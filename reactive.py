from __future__ import annotations
 
from make_dag     import TaskDAG, Task
from make_network import NetworkGraph, Processor
from dynamic_network import DynamicNetwork
from heft import calc_heft
 
 
def _sub_dag(dag: TaskDAG, remaining: set[int]) -> TaskDAG:
    """Sub-DAG containing only tasks in `remaining`, with their edges."""
    sub = TaskDAG()
    for tid in remaining:
        sub.nodes[tid] = Task(tid, dict(dag.nodes[tid].comp_costs))
    for tid in remaining:
        for child_id in dag.nodes[tid].children:
            if child_id in remaining:
                sub.add_edge(tid, child_id, comm_cost=dag.edges[(tid, child_id)])
    return sub
 
 
def _rebase(
    raw:        dict[int, tuple],   # output of calc_heft on sub-DAG (times start at 0)
    sub:        TaskDAG,
    orig_dag:   TaskDAG,
    fixed:      dict[int, tuple],   # committed tasks
    snapshot:   NetworkGraph,
    event_time: float,
) -> dict[int, tuple]:
    """
    Shift calc_heft's zero-based times forward so that:
      1. Nothing starts before event_time.
      2. Each task whose parent is committed starts after
         parent_finish + comm_cost.
      3. Each task whose parent is in the sub-DAG starts after
         that parent's (rebased) finish.
    """
    topo   = list(reversed(sub._topological_sort()))  # roots first
    rebased: dict[int, tuple] = {}
 
    for tid in topo:
        proc_id, raw_start, raw_finish = raw[tid]
        duration = raw_finish - raw_start
 
        # Floor = must not start before event_time
        floor = event_time
 
        # Cross-boundary: parent is a committed (fixed) task
        for parent_id in orig_dag.nodes[tid].parents:
            if parent_id in fixed:
                parent_proc, _, parent_finish = fixed[parent_id]
                data_size = orig_dag.edges[(parent_id, tid)]
                comm = snapshot.comm_cost(parent_proc, proc_id, data_size,
                                          fallback_bandwidth=None)
                if comm == float('inf'):
                    comm = 0.0
                floor = max(floor, parent_finish + comm)
 
        # Intra-sub-DAG: parent is also being rebased
        for parent_id in sub.nodes[tid].parents:
            if parent_id in rebased:
                floor = max(floor, rebased[parent_id][2])  # parent's rebased finish
 
        # Shift: the raw schedule may have already placed this task after
        # some internal waiting time.  Preserve that gap relative to the
        # latest rebased parent finish.
        new_start  = max(raw_start, floor)
        rebased[tid] = (proc_id, new_start, new_start + duration)
 
    return rebased
 
 
def simulate_reactive(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork,
) -> dict[int, tuple]:
    """
    Event-driven reactive scheduler.  Calls calc_heft() from scratch
    on remaining tasks at every processor topology change.
 
    Returns: {task_id: (proc_id, actual_start, actual_finish)}
    """
    # Only react to processor-set changes, not bandwidth fluctuations
    topology_events: list[float] = []
    prev_procs: set | None = None
    for ts, net in sorted(dynamic_net.snapshots, key=lambda x: x[0]):
        curr_procs = set(net.processors.keys())
        if prev_procs is None or curr_procs != prev_procs:
            topology_events.append(ts)
            prev_procs = curr_procs
 
    # Initial plan on the full base network
    current_plan: dict[int, tuple] = calc_heft(dag, network)
    actual:       dict[int, tuple] = {}
 
    for event_time in topology_events:
        snapshot = dynamic_net.pred_net_func(event_time)
 
        # Commit tasks whose planned start ≤ event_time
        for task_id, (proc_id, start, finish) in list(current_plan.items()):
            if start <= event_time and task_id not in actual:
                actual[task_id] = (proc_id, start, finish)
 
        remaining = set(dag.nodes) - set(actual)
        if not remaining or not snapshot.processors:
            continue
 
        # Build sub-DAG and call calc_heft — this is the reactive reschedule
        sub     = _sub_dag(dag, remaining)
        raw     = calc_heft(sub, snapshot)
 
        # Rebase times to respect committed-task finish times and event_time
        shifted = _rebase(raw, sub, dag, actual, snapshot, event_time)
 
        current_plan = {**actual, **shifted}
 
    # Commit anything left after the last event
    for task_id, entry in current_plan.items():
        if task_id not in actual:
            actual[task_id] = entry
 
    return actual