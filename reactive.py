from __future__ import annotations

from make_dag        import TaskDAG
from make_network    import NetworkGraph
from dynamic_network import DynamicNetwork

INF = float('inf')


class SchedulingError(RuntimeError):
    """Raised when tasks can't be placed. .issues = {task_id: reason}."""
    def __init__(self, message: str, issues: dict[int, str]):
        super().__init__(message)
        self.issues = issues


# ─────────────────────────────────────────────────────────────────────────────
# HEFT on the CURRENT snapshot, continuing from what has already happened
# ─────────────────────────────────────────────────────────────────────────────

def heft_from_state(
    dag:         TaskDAG,
    snapshot:    NetworkGraph,
    dynamic_net: DynamicNetwork,
    remaining:   set[int],
    fixed:       dict[int, tuple],
    busy_until:  dict[int, float],
    now:         float,
) -> tuple[dict[int, tuple], dict[int, str]]:
    """
    calc_heft for the tasks in `remaining`, starting at time `now`.

    Decisions (task order, processor choice) use ONLY the current snapshot:
      * ranks are computed on the snapshot
      * only processors in the snapshot are considered
      * EFTs are estimated with the snapshot's comm_cost
    Times are what would really happen: once a processor is chosen, the
    task's start/finish use dynamic_net.comm_cost_integrated() from each
    parent's finish time.

    fixed      — {task_id: (proc, start, finish)} finished or still running
    busy_until — {proc_id: finish of the task still running on it}

    Returns (plan, unplaced) where unplaced = {task_id: reason}. Unplaced
    tasks wait for the next network change.
    """
    # task prioritization
    ranks        = dag.compute_ranks(snapshot)
    sorted_tasks = sorted(remaining, key=lambda t: ranks[t], reverse=True)

    # processor selection
    schedule       = dict(fixed)                    # task_id: proc, start, finish
    proc_available = {p: max(now, busy_until.get(p, now)) for p in snapshot.processors}
    plan:     dict[int, tuple] = {}
    unplaced: dict[int, str]   = {}

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]

        waiting = [p for p in task.parents if p not in schedule]
        if waiting:
            unplaced[task_id] = f"waiting for parent T{waiting[0]}"
            continue

        best = None                                 # (estimated eft, proc, true start)
        for proc_id in snapshot.processors:
            if proc_id not in task.comp_costs:
                continue
            est_ready = true_ready = now
            for parent_id in task.parents:
                parent_proc, _, parent_eft = schedule[parent_id]
                data_size = dag.edges[(parent_id, task_id)]
                est_ready  = max(est_ready,  parent_eft + snapshot.comm_cost(
                    parent_proc, proc_id, data_size))
                true_ready = max(true_ready, parent_eft + dynamic_net.comm_cost_integrated(
                    parent_proc, proc_id, data_size, parent_eft))
            if true_ready == INF:
                continue                            # data can't be sent there

            est_eft = max(est_ready, proc_available[proc_id]) + task.comp_costs[proc_id]
            if best is None or est_eft < best[0]:
                best = (est_eft, proc_id, max(true_ready, proc_available[proc_id]))

        if best is None:
            unplaced[task_id] = "its input data can't reach any current processor"
            continue

        _, proc_id, start = best
        finish = start + task.comp_costs[proc_id]
        schedule[task_id]       = (proc_id, start, finish)
        plan[task_id]           = (proc_id, start, finish)
        proc_available[proc_id] = finish

    return plan, unplaced


# ─────────────────────────────────────────────────────────────────────────────
# Reactive simulation: re-run HEFT at every network change
# ─────────────────────────────────────────────────────────────────────────────

def simulate_reactive(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork,
    trace_file:  str | None = None,
) -> dict[int, tuple]:
    """
    At t=0 and at every network change (any snapshot time):
      1. Work out what has happened since the last change:
           finished (finish <= t)                       -> kept
           running  (start < t < finish), proc present -> kept; proc busy
           running  on a processor that left           -> KILLED, re-planned
           finished on a processor that left, when a child that hasn't
           started has no current processor all its input data can reach
                                                        -> output LOST, re-run
           not started (start >= t)                     -> re-planned
      2. Run HEFT on everything that isn't kept, using only the current
         snapshot (heft_from_state).

    Tasks that can't be placed wait for the next change. After the last
    change the network is fixed, so the current plan runs as written.

    Returns {task_id: (proc_id, start, finish)} — the run that completed.
    Raises SchedulingError (a RuntimeError) if some task can never be
    placed (no fallback); its .issues maps each task to the reason.
    """
    events = sorted({0.0} | {ts for ts, _ in dynamic_net.snapshots if ts >= 0})

    plan:     dict[int, tuple] = {}
    unplaced: dict[int, str]   = {}
    lines:    list[str]        = []

    for now in events:
        snapshot = dynamic_net.pred_net_func(now)
        present  = set(snapshot.processors)

        # ── 1. what has happened up to `now` ─────────────────────────────
        fixed:      dict[int, tuple] = {}
        busy_until: dict[int, float] = {}
        killed:     list[int]        = []
        for tid, (proc, start, finish) in plan.items():
            if finish <= now:
                fixed[tid] = (proc, start, finish)
            elif start < now:
                if proc in present:
                    fixed[tid]       = (proc, start, finish)
                    busy_until[proc] = max(busy_until.get(proc, now), finish)
                else:
                    killed.append(tid)

        # Output lost: a task that hasn't started needs data from its kept
        # parents, but there is no current processor that ALL of that data
        # can reach (comm_cost_integrated is infinite for at least one
        # parent). Its parents that finished on a processor that has left are
        # re-run. Repeat, since re-running a task can make its own parents'
        # outputs needed again.
        def has_target(child: int) -> bool:
            kept = [p for p in dag.nodes[child].parents if p in fixed]
            return any(
                q in dag.nodes[child].comp_costs and all(
                    dynamic_net.comm_cost_integrated(
                        fixed[p][0], q, dag.edges[(p, child)], fixed[p][2]) != INF
                    for p in kept)
                for q in present)

        lost: list[int] = []
        changed = True
        while changed:
            changed = False
            for child in dag.nodes:
                if child in fixed or not any(p in fixed for p in dag.nodes[child].parents):
                    continue
                if has_target(child):
                    continue
                for p in dag.nodes[child].parents:
                    if p in fixed and fixed[p][2] <= now and fixed[p][0] not in present:
                        del fixed[p]
                        lost.append(p)
                        changed = True

        # ── 2. re-plan everything else on the current snapshot ───────────
        remaining = set(dag.nodes) - set(fixed)
        if remaining and present:
            new_plan, unplaced = heft_from_state(
                dag, snapshot, dynamic_net, remaining, fixed, busy_until, now)
        else:
            new_plan, unplaced = {}, {t: "no processors in the network" for t in remaining}
        plan = {**fixed, **new_plan}

        if trace_file is not None:
            lines.append(f"t={now:.2f}  processors={sorted(present)}")
            if killed:
                lines.append(f"   killed (processor left): {[f'T{t}' for t in sorted(killed)]}")
            if lost:
                lines.append(f"   output lost, re-run:     {[f'T{t}' for t in sorted(lost)]}")
            if unplaced:
                lines.append(f"   waiting:                 {[f'T{t}' for t in sorted(unplaced)]}")
            for tid, (p, s, f) in sorted(new_plan.items(), key=lambda x: x[1][1]):
                lines.append(f"   T{tid} -> P{p + 1} [{s:.2f}-{f:.2f}]")

    if trace_file is not None:
        if plan:
            lines.append(f"\nMakespan: {max(f for _, _, f in plan.values()):.4f}")
        with open(trace_file, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')

    if unplaced:
        details = '; '.join(f"T{t}: {why}" for t, why in sorted(unplaced.items())[:5])
        raise SchedulingError(
            f"{len(unplaced)} task(s) could not be placed after the last network change ({details})",
            dict(unplaced))

    return plan