from __future__ import annotations
 
import math
from typing import Optional
 
from make_dag import TaskDAG
from make_network import NetworkGraph
 
 
# ── solver availability probe ──────────────────────────────────────────────
 
def _pulp_available() -> bool:
    try:
        import pulp  # noqa: F401
        return True
    except ImportError:
        return False
 
 
# ── critical-path lower bound (always available, O(n)) ────────────────────
 
def _cp_lower_bound(dag: TaskDAG, network: NetworkGraph) -> dict:
    """
    Fastest lower bound: longest path through DAG using each task's
    minimum computation cost across all processors, zero communication.
    Always valid; used as fallback when PuLP is unavailable.
    """
    from collections import deque
 
    in_degree = {tid: len(t.parents) for tid, t in dag.nodes.items()}
    children  = {tid: [] for tid in dag.nodes}
    for (src, dst) in dag.edges:
        children[src].append(dst)
 
    dist  = {}
    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    while queue:
        tid      = queue.popleft()
        min_cost = min(dag.nodes[tid].comp_costs[p] for p in network.processors
                       if p in dag.nodes[tid].comp_costs)
        dist[tid] = (
            max((dist[p] for p in dag.nodes[tid].parents), default=0.0) + min_cost
        )
        for child_id in children[tid]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)
 
    makespan = max(dist.values(), default=0.0)
 
    # Build a minimal schedule from the CP bound
    schedule = {}
    proc_avail = {p: 0.0 for p in network.processors}
    topo = _topo_sort(dag)
    for tid in topo:
        task = dag.nodes[tid]
        best_p = min(network.processors,
                     key=lambda p: task.comp_costs.get(p, float('inf')))
        ready = max(
            (schedule[par][2] for par in task.parents if par in schedule),
            default=0.0
        )
        start = max(ready, proc_avail[best_p])
        finish = start + task.comp_costs[best_p]
        schedule[tid] = (best_p, start, finish)
        proc_avail[best_p] = finish
 
    return {
        'makespan': makespan,
        'schedule': schedule,
        'gap': 0.0,
        'method': 'fallback_cp',
    }
 
 
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
 
 
# ── ILP / LP formulation ───────────────────────────────────────────────────
 
def _solve_ilp(
    dag: TaskDAG,
    network: NetworkGraph,
    relax: bool,
    time_limit: int,
) -> dict:
    """
    MILP formulation (Sinnen & Sousa 2004 / standard list-scheduling MILP).
 
    Variables
    ---------
    s[i]        continuous start time of task i
    C           continuous makespan (objective)
    x[i][k]     binary (or relaxed continuous in [0,1]): task i assigned to proc k
 
    Constraints
    -----------
    (A) assignment:   sum_k x[i][k] = 1   for all i
    (B) makespan:     C >= s[i] + sum_k x[i][k]*w[i][k]   for all i
    (C) precedence:   s[j] >= s[i] + sum_k x[i][k]*w[i][k]
                             + sum_{k1,k2} x[i][k1]*x[j][k2]*c[i][j][k1][k2]
                      (linearised — see below)
    (D) non-overlap:  s[j] >= s[i] + sum_k x[i][k]*w[i][k]  OR
                      s[i] >= s[j] + sum_k x[j][k]*w[j][k]
                      (disjunctive — big-M linearisation)
 
    Linearisation of bilinear x[i][k1]*x[j][k2]
    ---------------------------------------------
    We introduce y[i][j][k1][k2] = x[i][k1] * x[j][k2] and add the
    standard McCormick envelopes.  This blows up variable count for large
    instances, so we cap at MAX_TASKS × MAX_PROCS.
    """
    import pulp
 
    tasks     = list(dag.nodes.keys())
    procs     = list(network.processors.keys())
    n_tasks   = len(tasks)
    n_procs   = len(procs)
 
    # Tight upper bound on makespan for big-M
    BIG_M = sum(
        max(dag.nodes[t].comp_costs.get(p, 0) for p in procs)
        for t in tasks
    ) + sum(
        dag.edges[e] / min(network.bandwidth.values())
        for e in dag.edges
    )
 
    prob = pulp.LpProblem("DAG_Scheduling", pulp.LpMinimize)
    cat  = 'Continuous' if relax else 'Binary'
 
    # ── decision variables ─────────────────────────────────────────────────
    x = {
        (i, k): pulp.LpVariable(f"x_{i}_{k}", 0, 1, cat=cat)
        for i in tasks for k in procs
    }
    s = {i: pulp.LpVariable(f"s_{i}", 0) for i in tasks}
    C = pulp.LpVariable("C", 0)
 
    # McCormick auxiliary: y[i,j,k1,k2] ≈ x[i,k1]*x[j,k2]  (only for edges)
    y = {}
    for (src, dst) in dag.edges:
        for k1 in procs:
            for k2 in procs:
                y[(src, dst, k1, k2)] = pulp.LpVariable(
                    f"y_{src}_{dst}_{k1}_{k2}", 0, 1
                )
 
    # ── objective ──────────────────────────────────────────────────────────
    prob += C
 
    # ── (A) assignment ─────────────────────────────────────────────────────
    for i in tasks:
        prob += pulp.lpSum(x[i, k] for k in procs) == 1
 
    # ── (B) makespan ───────────────────────────────────────────────────────
    for i in tasks:
        w_expr = pulp.lpSum(dag.nodes[i].comp_costs.get(k, BIG_M) * x[i, k] for k in procs)
        prob += C >= s[i] + w_expr
 
    # ── (C) precedence + McCormick for communication ───────────────────────
    for (src, dst) in dag.edges:
        data = dag.edges[(src, dst)]
        w_src = pulp.lpSum(dag.nodes[src].comp_costs.get(k, BIG_M) * x[src, k] for k in procs)
 
        # communication term: sum_{k1,k2} comm[k1][k2] * y[src,dst,k1,k2]
        comm_expr = pulp.lpSum(
            (0.0 if k1 == k2 else
             data / network.bandwidth.get((k1, k2), 1e-6)) * y[(src, dst, k1, k2)]
            for k1 in procs for k2 in procs
        )
 
        prob += s[dst] >= s[src] + w_src + comm_expr
 
        # McCormick envelopes for each (k1, k2)
        for k1 in procs:
            for k2 in procs:
                yv = y[(src, dst, k1, k2)]
                xsk1 = x[src, k1]
                xdk2 = x[dst, k2]
                prob += yv >= xsk1 + xdk2 - 1
                prob += yv <= xsk1
                prob += yv <= xdk2
                prob += yv >= 0
 
    # ── (D) non-overlap (disjunctive) — only for tasks on same processor ──
    # We use the big-M disjunction:
    #   s[j] >= s[i] + w[i] - BIG_M*(1-z[i,j,k]) - BIG_M*(1-x[i,k]) - BIG_M*(1-x[j,k])
    # where z[i,j,k]=1 means i runs before j on k.
    # This is O(n²×P) variables which we only add when n_tasks × n_procs is small.
    MAX_DISJUNCTIVE = 600   # beyond this, skip non-overlap (LP relaxation becomes looser)
 
    if n_tasks * n_procs <= MAX_DISJUNCTIVE:
        z = {}
        for idx, i in enumerate(tasks):
            for j in tasks[idx+1:]:
                for k in procs:
                    zv = pulp.LpVariable(f"z_{i}_{j}_{k}", 0, 1,
                                         cat='Continuous' if relax else 'Binary')
                    z[(i, j, k)] = zv
                    wi = pulp.lpSum(dag.nodes[i].comp_costs.get(k2, BIG_M) * x[i, k2] for k2 in procs)
                    wj = pulp.lpSum(dag.nodes[j].comp_costs.get(k2, BIG_M) * x[j, k2] for k2 in procs)
                    # i before j
                    prob += s[j] >= s[i] + wi - BIG_M*(1 - zv) - BIG_M*(1 - x[i, k]) - BIG_M*(1 - x[j, k])
                    # j before i
                    prob += s[i] >= s[j] + wj - BIG_M*zv       - BIG_M*(1 - x[i, k]) - BIG_M*(1 - x[j, k])
 
    # ── solve ──────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)
 
    status = pulp.LpStatus[prob.status]
    if status not in ('Optimal', 'Not Solved'):
        # infeasible or error — fall back
        return None
 
    makespan_val = pulp.value(C)
    if makespan_val is None:
        return None
 
    # ── extract schedule ───────────────────────────────────────────────────
    schedule = {}
    for i in tasks:
        assigned_proc = max(procs, key=lambda k: pulp.value(x[i, k]) or 0.0)
        start_val  = pulp.value(s[i]) or 0.0
        comp_cost  = dag.nodes[i].comp_costs.get(assigned_proc, 0.0)
        finish_val = start_val + comp_cost
        schedule[i] = (assigned_proc, start_val, finish_val)
 
    # MIP gap
    gap = 0.0
    try:
        gap = abs(prob.solver.solverModel.bestBound - makespan_val) / (abs(makespan_val) + 1e-9)
    except Exception:
        pass
 
    return {
        'makespan': makespan_val,
        'schedule': schedule,
        'gap': gap,
        'method': 'LP_relaxation' if relax else 'MIP',
    }
 
 
# ── public entry point ─────────────────────────────────────────────────────
 
# Thresholds for auto-selection of method
_MIP_MAX_TASKS  = 60
_MIP_MAX_PROCS  = 8
_LP_MAX_TASKS   = 200
 
 
def calc_oracle_ilp(
    dag: TaskDAG,
    network: NetworkGraph,
    use_lp_relaxation: bool = False,
    time_limit: int = 60,
) -> dict:
    """
    Compute an oracle-optimal (or best-lower-bound) makespan.
 
    Parameters
    ----------
    dag               : TaskDAG
    network           : NetworkGraph  (static, perfect knowledge)
    use_lp_relaxation : if True, always solve the LP relaxation (faster,
                        still a valid lower bound)
    time_limit        : seconds for the MIP solver
 
    Returns
    -------
    dict with keys:
        makespan  float
        schedule  {task_id: (proc_id, start, finish)}
        gap       float   (MIP optimality gap; 0 for LP/CP)
        method    str     ('MIP' | 'LP_relaxation' | 'fallback_cp')
    """
    if not _pulp_available():
        print("  [oracle] PuLP not found — using CP lower bound as fallback.")
        return _cp_lower_bound(dag, network)
 
    n_tasks = len(dag.nodes)
    n_procs = len(network.processors)
 
    # Auto-select method based on problem size
    if use_lp_relaxation:
        relax = True
    elif n_tasks <= _MIP_MAX_TASKS and n_procs <= _MIP_MAX_PROCS:
        relax = False      # full MIP — tractable
    elif n_tasks <= _LP_MAX_TASKS:
        relax = True       # LP relaxation — still a lower bound
        print(f"  [oracle] Problem too large for MIP ({n_tasks}t × {n_procs}p) "
              f"— using LP relaxation.")
    else:
        print(f"  [oracle] Problem too large for LP ({n_tasks}t × {n_procs}p) "
              f"— using CP lower bound.")
        return _cp_lower_bound(dag, network)
 
    result = _solve_ilp(dag, network, relax=relax, time_limit=time_limit)
 
    if result is None:
        print("  [oracle] Solver failed — falling back to CP lower bound.")
        return _cp_lower_bound(dag, network)
 
    return result