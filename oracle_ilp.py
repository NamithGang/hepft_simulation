from __future__ import annotations

from make_dag     import TaskDAG
from make_network import NetworkGraph


# ── solver availability ────────────────────────────────────────────────────

def _pulp_available() -> bool:
    try:
        import pulp  # noqa: F401
        return True
    except ImportError:
        return False


# ── critical-path lower bound (no solver required) ─────────────────────────

def _cp_lower_bound(dag: TaskDAG, network: NetworkGraph) -> dict:
    """
    Longest path through the DAG using each task's minimum computation
    cost and zero communication.  Always a valid lower bound; O(n).
    """
    from collections import deque

    children  = {tid: [] for tid in dag.nodes}
    in_degree = {tid: len(t.parents) for tid, t in dag.nodes.items()}
    for (src, dst) in dag.edges:
        children[src].append(dst)

    dist  = {}
    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    while queue:
        tid      = queue.popleft()
        min_cost = min(dag.nodes[tid].comp_costs[p] for p in network.processors
                       if p in dag.nodes[tid].comp_costs)
        dist[tid] = max((dist[p] for p in dag.nodes[tid].parents), default=0.0) + min_cost
        for child_id in children[tid]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    makespan = max(dist.values(), default=0.0)

    # Build a greedy schedule so callers get a usable assignment
    schedule   = {}
    proc_avail = {p: 0.0 for p in network.processors}
    for tid in _topo_sort(dag):
        task   = dag.nodes[tid]
        best_p = min(network.processors, key=lambda p: task.comp_costs.get(p, float('inf')))
        ready  = max((schedule[par][2] for par in task.parents if par in schedule), default=0.0)
        start  = max(ready, proc_avail[best_p])
        finish = start + task.comp_costs[best_p]
        schedule[tid]      = (best_p, start, finish)
        proc_avail[best_p] = finish

    return {'makespan': makespan, 'schedule': schedule, 'gap': 0.0, 'method': 'fallback_cp'}


def _topo_sort(dag: TaskDAG) -> list[int]:
    visited, order = set(), []

    def dfs(tid):
        visited.add(tid)
        for c in dag.nodes[tid].children:
            if c not in visited:
                dfs(c)
        order.append(tid)

    for tid in dag.nodes:
        if tid not in visited:
            dfs(tid)
    return list(reversed(order))


# ── MILP formulation ───────────────────────────────────────────────────────

def _solve_ilp(dag: TaskDAG, network: NetworkGraph, relax: bool, time_limit: int) -> dict | None:
    """
    MILP formulation following Sinnen & Sousa (2004).

    Variables
    ---------
    x[i, k]          binary (or continuous if relax=True): task i on processor k
    s[i]              continuous start time of task i
    C                 continuous makespan (objective)
    y[i, j, k1, k2]  McCormick auxiliary: approximates x[i,k1] * x[j,k2]
    z[i, j, k]        binary ordering: task i runs before task j on processor k

    Constraints
    -----------
    (A) Each task assigned to exactly one processor.
    (B) Makespan >= finish time of every task.
    (C) Precedence: child starts after parent finishes + communication time.
        Communication is linearised via McCormick envelopes on y.
    (D) Non-overlap: for each processor, each pair of tasks is ordered.
        Uses big-M disjunction.  Only added when problem is small enough.
    """
    import pulp

    tasks = list(dag.nodes.keys())
    procs = list(network.processors.keys())
    cat   = 'Continuous' if relax else 'Binary'

    # Upper bound on makespan used in big-M constraints
    BIG_M = (
        sum(max(dag.nodes[t].comp_costs.get(p, 0) for p in procs) for t in tasks)
        + sum(dag.edges[e] / min(network.bandwidth.values()) for e in dag.edges)
    )

    prob = pulp.LpProblem("DAG_Scheduling", pulp.LpMinimize)

    x = {(i, k): pulp.LpVariable(f"x_{i}_{k}", 0, 1, cat=cat) for i in tasks for k in procs}
    s = {i:      pulp.LpVariable(f"s_{i}", 0)                   for i in tasks}
    C =          pulp.LpVariable("C", 0)

    # McCormick auxiliaries — one per directed edge and processor pair
    y = {
        (src, dst, k1, k2): pulp.LpVariable(f"y_{src}_{dst}_{k1}_{k2}", 0, 1)
        for (src, dst) in dag.edges
        for k1 in procs
        for k2 in procs
    }

    prob += C   # objective: minimise makespan

    # (A) assignment
    for i in tasks:
        prob += pulp.lpSum(x[i, k] for k in procs) == 1

    # (B) makespan
    for i in tasks:
        prob += C >= s[i] + pulp.lpSum(dag.nodes[i].comp_costs.get(k, BIG_M) * x[i, k] for k in procs)

    # (C) precedence with linearised communication
    for (src, dst) in dag.edges:
        data  = dag.edges[(src, dst)]
        w_src = pulp.lpSum(dag.nodes[src].comp_costs.get(k, BIG_M) * x[src, k] for k in procs)
        comm  = pulp.lpSum(
            (0.0 if k1 == k2 else data / network.bandwidth.get((k1, k2), 1e-6))
            * y[(src, dst, k1, k2)]
            for k1 in procs for k2 in procs
        )
        prob += s[dst] >= s[src] + w_src + comm

        # McCormick envelopes: y[src,dst,k1,k2] approximates x[src,k1] * x[dst,k2]
        for k1 in procs:
            for k2 in procs:
                yv = y[(src, dst, k1, k2)]
                prob += yv >= x[src, k1] + x[dst, k2] - 1
                prob += yv <= x[src, k1]
                prob += yv <= x[dst, k2]

    # (D) non-overlap — only when the problem is small enough to be tractable
    if len(tasks) * len(procs) <= 600:
        for idx, i in enumerate(tasks):
            for j in tasks[idx + 1:]:
                for k in procs:
                    zv = pulp.LpVariable(f"z_{i}_{j}_{k}", 0, 1,
                                         cat='Continuous' if relax else 'Binary')
                    wi = pulp.lpSum(dag.nodes[i].comp_costs.get(k2, BIG_M) * x[i, k2] for k2 in procs)
                    wj = pulp.lpSum(dag.nodes[j].comp_costs.get(k2, BIG_M) * x[j, k2] for k2 in procs)
                    prob += s[j] >= s[i] + wi - BIG_M * (1 - zv) - BIG_M * (1 - x[i, k]) - BIG_M * (1 - x[j, k])
                    prob += s[i] >= s[j] + wj - BIG_M * zv       - BIG_M * (1 - x[i, k]) - BIG_M * (1 - x[j, k])

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    if pulp.LpStatus[prob.status] not in ('Optimal', 'Not Solved'):
        return None

    makespan_val = pulp.value(C)
    if makespan_val is None:
        return None

    # Extract assignment: pick the processor with the highest x[i,k] value
    schedule = {}
    for i in tasks:
        proc_id   = max(procs, key=lambda k: pulp.value(x[i, k]) or 0.0)
        start_val = pulp.value(s[i]) or 0.0
        schedule[i] = (proc_id, start_val, start_val + dag.nodes[i].comp_costs.get(proc_id, 0.0))

    gap = 0.0
    try:
        gap = abs(prob.solver.solverModel.bestBound - makespan_val) / (abs(makespan_val) + 1e-9)
    except Exception:
        pass

    return {'makespan': makespan_val, 'schedule': schedule, 'gap': gap,
            'method': 'LP_relaxation' if relax else 'MIP'}


# ── size thresholds for method selection ───────────────────────────────────

_MIP_MAX_TASKS = 60
_MIP_MAX_PROCS = 8
_LP_MAX_TASKS  = 200


# ── public entry point ─────────────────────────────────────────────────────

def calc_oracle_ilp(
    dag:               TaskDAG,
    network:           NetworkGraph,
    use_lp_relaxation: bool = False,
    time_limit:        int  = 60,
) -> dict:
    """
    Compute an oracle-optimal (or tightest available) makespan lower bound.

    Returns a dict with keys:
        makespan  float
        schedule  {task_id: (proc_id, start, finish)}
        gap       float   (MIP optimality gap; 0.0 for LP / CP)
        method    str     ('MIP' | 'LP_relaxation' | 'fallback_cp')
    """
    if not _pulp_available():
        print("  [oracle] PuLP not found — using CP lower bound.")
        return _cp_lower_bound(dag, network)

    n_tasks = len(dag.nodes)
    n_procs = len(network.processors)

    if use_lp_relaxation:
        relax = True
    elif n_tasks <= _MIP_MAX_TASKS and n_procs <= _MIP_MAX_PROCS:
        relax = False
    elif n_tasks <= _LP_MAX_TASKS:
        relax = True
        print(f"  [oracle] {n_tasks}t × {n_procs}p too large for MIP — using LP relaxation.")
    else:
        print(f"  [oracle] {n_tasks}t × {n_procs}p too large for LP — using CP lower bound.")
        return _cp_lower_bound(dag, network)

    result = _solve_ilp(dag, network, relax=relax, time_limit=time_limit)

    if result is None:
        print("  [oracle] Solver failed — falling back to CP lower bound.")
        return _cp_lower_bound(dag, network)

    return result