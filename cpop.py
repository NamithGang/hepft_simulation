from make_dag import TaskDAG
from make_network import NetworkGraph
 
 
# ── helpers ────────────────────────────────────────────────────────────────
 
def _avg_comp(task, network: NetworkGraph) -> float:
    return sum(task.comp_costs[p] for p in network.processors) / len(network.processors)
 
 
def _avg_comm(dag: TaskDAG, src_id: int, dst_id: int, network: NetworkGraph) -> float:
    bw_vals = list(network.bandwidth.values())
    avg_bw = sum(bw_vals) / len(bw_vals)
    return dag.edges[(src_id, dst_id)] / avg_bw
 
 
# ── rank computation ────────────────────────────────────────────────────────
 
def _compute_rank_u(dag: TaskDAG, network: NetworkGraph) -> dict[int, float]:
    """Upward rank (same as HEFT rank)."""
    rank_u: dict[int, float] = {}
    # _topological_sort() puts children before parents (leaves first) — correct for upward rank
    for task_id in dag._topological_sort():
        task = dag.nodes[task_id]
        w = _avg_comp(task, network)
        if not task.children:
            rank_u[task_id] = w
        else:
            rank_u[task_id] = w + max(
                _avg_comm(dag, task_id, c, network) + rank_u[c]
                for c in task.children
            )
    return rank_u
 
 
def _compute_rank_d(dag: TaskDAG, network: NetworkGraph) -> dict[int, float]:
    """Downward rank — longest path from entry to (but not including) this task."""
    rank_d: dict[int, float] = {}
    # reversed gives parents before children (roots first) — correct for downward rank
    for task_id in reversed(list(dag._topological_sort())):
        task = dag.nodes[task_id]
        if not task.parents:
            rank_d[task_id] = 0.0
        else:
            rank_d[task_id] = max(
                rank_d[p] + _avg_comp(dag.nodes[p], network)
                + _avg_comm(dag, p, task_id, network)
                for p in task.parents
            )
    return rank_d
 
 
# ── critical path ───────────────────────────────────────────────────────────
 
def _critical_path(dag: TaskDAG, rank_u: dict, rank_d: dict) -> set[int]:
    """
    CP = set of tasks whose (rank_u + rank_d) equals the maximum priority.
    The max priority is always rank_u of any entry task (rank_d = 0 there).
    """
    max_priority = max(rank_u[t] + rank_d[t] for t in dag.nodes)
    # small epsilon to handle floating-point ties
    eps = 1e-9
    return {t for t in dag.nodes if abs(rank_u[t] + rank_d[t] - max_priority) < eps}
 
 
def _choose_critical_processor(
    dag: TaskDAG,
    network: NetworkGraph,
    cp_tasks: set[int],
    rank_u: dict,
) -> int:
    """
    For each candidate processor, estimate total execution time if every
    CP task were pinned to it (ignoring intra-CP communication, which is
    zero when tasks share a processor).  Return the processor with the
    smallest total CP execution time.
    """
    best_proc, best_time = None, float('inf')
    # topological order restricted to CP tasks
    cp_topo = [t for t in dag._topological_sort() if t in cp_tasks]
 
    for proc_id in network.processors:
        total = sum(dag.nodes[t].comp_costs[proc_id] for t in cp_topo)
        if total < best_time:
            best_time, best_proc = total, proc_id
 
    return best_proc
 
 
# ── main scheduler ──────────────────────────────────────────────────────────
 
def calc_cpop(dag: TaskDAG, network: NetworkGraph) -> dict:
    """
    Returns schedule: {task_id: (proc_id, start_time, finish_time)}
    """
    rank_u = _compute_rank_u(dag, network)
    rank_d = _compute_rank_d(dag, network)
 
    cp_tasks = _critical_path(dag, rank_u, rank_d)
    cp_proc  = _choose_critical_processor(dag, network, cp_tasks, rank_u)
 
    # sort by decreasing priority (rank_u + rank_d)
    priority = {t: rank_u[t] + rank_d[t] for t in dag.nodes}
    sorted_tasks = sorted(dag.nodes, key=lambda t: priority[t], reverse=True)
 
    import heapq
 
    schedule:       dict[int, tuple] = {}
    proc_available: dict[int, float] = {p: 0.0 for p in network.processors}
 
    # Track how many parents each task has scheduled
    parents_done = {t: 0 for t in dag.nodes}
    n_parents    = {t: len(dag.nodes[t].parents) for t in dag.nodes}
 
    # Priority queue: (-priority, task_id)  — max-heap via negation
    ready: list = []
    for t in dag.nodes:
        if n_parents[t] == 0:
            heapq.heappush(ready, (-priority[t], t))
 
    while ready:
        _, task_id = heapq.heappop(ready)
        task = dag.nodes[task_id]
 
        if task_id in cp_tasks:
            # ── CP task: must go to the critical processor ──────────────
            proc_id = cp_proc
            ready_time = 0.0
            for parent_id in task.parents:
                parent_proc, _, parent_eft = schedule[parent_id]
                data_size = dag.edges[(parent_id, task_id)]
                comm = network.comm_cost(parent_proc, proc_id, data_size)
                ready_time = max(ready_time, parent_eft + comm)
 
            est = max(ready_time, proc_available[proc_id])
            eft = est + task.comp_costs[proc_id]
 
        else:
            # ── non-CP task: EFT-minimising selection (same as HEFT) ───
            best_proc, best_est, best_eft = None, None, float('inf')
 
            for proc_id in network.processors:
                ready_time = 0.0
                for parent_id in task.parents:
                    parent_proc, _, parent_eft = schedule[parent_id]
                    data_size = dag.edges[(parent_id, task_id)]
                    comm = network.comm_cost(parent_proc, proc_id, data_size)
                    ready_time = max(ready_time, parent_eft + comm)
 
                est = max(ready_time, proc_available[proc_id])
                eft = est + task.comp_costs[proc_id]
 
                if eft < best_eft:
                    best_eft, best_est, best_proc = eft, est, proc_id
 
            proc_id, est, eft = best_proc, best_est, best_eft
 
        schedule[task_id]        = (proc_id, est, eft)
        proc_available[proc_id]  = eft
 
        # Unlock children whose parents are all done
        for child_id in task.children:
            parents_done[child_id] += 1
            if parents_done[child_id] == n_parents[child_id]:
                heapq.heappush(ready, (-priority[child_id], child_id))
 
    return schedule
 