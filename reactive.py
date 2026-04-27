from __future__ import annotations

from make_dag import TaskDAG
from make_network import NetworkGraph
from dynamic_network import DynamicNetwork
from heft import calc_heft


# ── internal HEFT on an arbitrary sub-DAG ──────────────────────────────────

def _heft_on_snapshot(
    dag: TaskDAG,
    snapshot: NetworkGraph,
    remaining_task_ids: set[int],
    fixed: dict[int, tuple],          # task_id -> (proc_id, start, finish)
    proc_occupied_until: dict[int, float],
) -> dict[int, tuple]:
    """
    Re-run HEFT on only the *remaining* (unstarted) tasks, using:
      - `snapshot`            — current network state
      - `fixed`               — already-completed tasks (immutable)
      - `proc_occupied_until` — when each processor becomes free again

    Returns a full planned schedule for the remaining tasks.
    """
    bw_vals = list(snapshot.bandwidth.values())
    if not bw_vals:
        return {}
    avg_bw = sum(bw_vals) / len(bw_vals)

    rank_u: dict[int, float] = {}

    full_topo      = dag._topological_sort()
    remaining_topo = [t for t in full_topo if t in remaining_task_ids]

    for task_id in remaining_topo:
        task = dag.nodes[task_id]
        if task_id not in snapshot.processors and len(snapshot.processors) == 0:
            continue
        avail_procs = list(snapshot.processors.keys())
        if not avail_procs:
            break
        avg_comp = (
            sum(task.comp_costs.get(p, float('inf')) for p in avail_procs)
            / len(avail_procs)
        )
        remaining_children = [c for c in task.children if c in remaining_task_ids]
        if not remaining_children:
            rank_u[task_id] = avg_comp
        else:
            rank_u[task_id] = avg_comp + max(
                (dag.edges[(task_id, c)] / avg_bw) + rank_u.get(c, 0.0)
                for c in remaining_children
            )

    sorted_remaining = sorted(
        remaining_task_ids,
        key=lambda t: rank_u.get(t, 0.0),
        reverse=True,
    )

    planned:    dict[int, tuple] = {}
    proc_avail: dict[int, float] = dict(proc_occupied_until)

    for task_id in sorted_remaining:
        task        = dag.nodes[task_id]
        avail_procs = list(snapshot.processors.keys())
        if not avail_procs:
            continue

        best_proc, best_est, best_eft = None, None, float('inf')

        for proc_id in avail_procs:
            ready_time = 0.0

            for parent_id in task.parents:
                if parent_id in fixed:
                    parent_proc, _, parent_eft = fixed[parent_id]
                elif parent_id in planned:
                    parent_proc, _, parent_eft = planned[parent_id]
                else:
                    parent_proc, parent_eft = proc_id, 0.0

                data_size = dag.edges[(parent_id, task_id)]
                comm = snapshot.comm_cost(parent_proc, proc_id, data_size,
                                          fallback_bandwidth=None)
                if comm == float('inf'):
                    comm = 0.0
                ready_time = max(ready_time, parent_eft + comm)

            est = max(ready_time, proc_avail.get(proc_id, 0.0))
            eft = est + task.comp_costs.get(proc_id, float('inf'))

            if eft < best_eft:
                best_eft, best_est, best_proc = eft, est, proc_id

        if best_proc is None:
            continue

        planned[task_id]      = (best_proc, best_est, best_eft)
        proc_avail[best_proc] = best_eft

    return planned


# ── main simulation driver ─────────────────────────────────────────────────

def simulate_reactive(
    dag: TaskDAG,
    network: NetworkGraph,
    dynamic_net: DynamicNetwork,
) -> dict[int, tuple]:
    """
    Event-driven reactive simulation.

    At each network-change event we check whether the *current planned*
    assignments are still valid and reschedule unstarted tasks if the
    network has changed in a way that would affect them.

    Timeout is enforced externally by run_demo.py's run_with_timeout(),
    which runs this function in a daemon thread and abandons it if it
    exceeds its budget.

    Returns: {task_id: (proc_id, actual_start, actual_finish)}
    """
    event_times  = sorted({ts for ts, _ in dynamic_net.snapshots})
    initial_plan = calc_heft(dag, network)

    actual:      dict[int, tuple] = {}
    current_plan = dict(initial_plan)
    current_time = 0.0

    for event_time in event_times:
        snapshot = dynamic_net.pred_net_func(event_time)

        for task_id, (proc_id, start, finish) in list(current_plan.items()):
            if start <= event_time and task_id not in actual:
                actual[task_id] = (proc_id, start, finish)

        remaining = set(dag.nodes) - set(actual)
        if not remaining:
            break

        proc_occupied: dict[int, float] = {p: 0.0 for p in snapshot.processors}
        for _, (proc_id, _, finish) in actual.items():
            if proc_id in proc_occupied:
                proc_occupied[proc_id] = max(proc_occupied[proc_id], finish)

        new_plan = _heft_on_snapshot(
            dag, snapshot, remaining, actual, proc_occupied
        )

        current_plan = {**{t: v for t, v in actual.items()}, **new_plan}
        current_time = event_time

    for task_id, entry in current_plan.items():
        if task_id not in actual:
            actual[task_id] = entry

    return actual