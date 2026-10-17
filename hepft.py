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


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — TASK RANKING
# ═════════════════════════════════════════════════════════════════════════════

def rank_tasks(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork | None = None,
) -> list[int]:
    """
    Return task IDs in scheduling order (highest upward rank first).

    Static:  ranks are computed on `network`.
    Dynamic: ranks are computed on a time-weighted AVERAGE of the snapshots
             covering [0, estimated makespan], so a single mispredicted
             snapshot can't distort the priority order.
             - Every processor present at some point in the window is kept.
             - Bandwidth is averaged over the whole window, so a link that
               only exists part of the time counts as proportionally slower.
             - Raises ValueError only if the window contains no processors.
    """
    if dynamic_net is None or not dag.nodes:
        ranks = dag.compute_ranks(network)
        return sorted(ranks, key=lambda t: ranks[t], reverse=True)

    # Window end = largest static rank + one average task's compute cost
    static_ranks = dag.compute_ranks(network)
    avg_comp = sum(
        sum(t.comp_costs.values()) / len(t.comp_costs) for t in dag.nodes.values()
    ) / len(dag.nodes)
    t_end = max(static_ranks.values()) + avg_comp

    # Each snapshot is active from its timestamp until the next one.
    # Collect (weight, snapshot) for the part of each that falls in [0, t_end].
    snaps    = sorted(dynamic_net.snapshots, key=lambda x: x[0])
    segments = []
    for i, (ts, snap) in enumerate(snaps):
        next_ts = snaps[i + 1][0] if i + 1 < len(snaps) else INF
        weight  = min(t_end, next_ts) - max(0.0, ts)
        if weight > 0:
            segments.append((weight, snap))

    procs = set().union(*(s.processors for _, s in segments)) if segments else set()
    if not procs:
        raise ValueError(
            f"No processor is present at any point in the ranking window [0, {t_end:.2f}]")

    total    = sum(w for w, _ in segments)
    rank_net = NetworkGraph()
    # Processor objects come from the latest segment that contains each one
    rank_net.processors = {
        p: next(s.processors[p] for _, s in reversed(segments) if p in s.processors)
        for p in procs
    }

    if isinstance(segments[-1][1].bandwidth, (int, float)):
        rank_net.bandwidth = sum(w * s.bandwidth for w, s in segments) / total
    else:
        weighted: dict[tuple, float] = {}
        for w, s in segments:
            for pair, bw in s.bandwidth.items():
                weighted[pair] = weighted.get(pair, 0.0) + bw * w
        rank_net.bandwidth = {pair: wb / total for pair, wb in weighted.items()}

    ranks = dag.compute_ranks(rank_net)
    return sorted(ranks, key=lambda t: ranks[t], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — DECIDING WHICH PROCESSORS TO INCLUDE
# ═════════════════════════════════════════════════════════════════════════════

def all_processors(network: NetworkGraph, dynamic_net: DynamicNetwork | None) -> set[int]:
    """
    Every processor that exists at any time: the starting network's plus any
    that join in a later snapshot.
    """
    procs = set(network.processors)
    if dynamic_net is not None:
        for _, snap in dynamic_net.snapshots:
            procs |= set(snap.processors)
    return procs


def processor_stays_up(
    dynamic_net: DynamicNetwork,
    proc_id:     int,
    start:       float,
    finish:      float,
) -> bool:
    """
    True if proc_id is present for the whole run [start, finish): at the
    start and at every snapshot change before the finish. Dropping out
    exactly at the finish is fine — the task is already done.
    """
    if not dynamic_net.pred_net_func(start).has_processor(proc_id):
        return False
    t = start
    while True:
        t = dynamic_net.next_snapshot_time(t)
        if t == INF or t >= finish:
            return True
        if not dynamic_net.pred_net_func(t).has_processor(proc_id):
            return False


def output_can_leave(
    dag:         TaskDAG,
    dynamic_net: DynamicNetwork,
    task_id:     int,
    proc_id:     int,
    finish:      float,
    candidates:  set[int],
) -> bool:
    """
    True if, when task_id finishes on proc_id at `finish`, the data for each
    of its children can reach at least one processor that can run that
    child (has a cost for it and is up at some point from `finish` on).
    Prevents a task's output from being stranded on a processor that leaves.
    """
    def ever_up_from(q: int, t: float) -> bool:
        while True:
            if dynamic_net.pred_net_func(t).has_processor(q):
                return True
            t = dynamic_net.next_snapshot_time(t)
            if t == INF:
                return False

    alive = {q for q in candidates if ever_up_from(q, finish)}
    for child_id in dag.nodes[task_id].children:
        size  = dag.edges[(task_id, child_id)]
        costs = dag.nodes[child_id].comp_costs
        if not any(q in costs and
                   dynamic_net.comm_cost_integrated(proc_id, q, size, finish) != INF
                   for q in alive):
            return False
    return True


def choose_processor(
    dynamic_net: DynamicNetwork | None,
    options:     dict[int, tuple[float, float]],
    output_ok=lambda proc_id, finish: True,
) -> tuple[int, float, float] | None:
    """
    Pick the processor for one task.

    options : {proc_id: (est, duration)} for every reachable processor.

    1. At the current instant, include only processors that stay up for
       the whole run, and pick the smallest EFT.
    2. If none qualifies, move to the NEXT snapshot instant, push every
       start to at least that instant, and re-check. Repeat until a
       processor qualifies.

    Processors only appear/disappear at snapshot times, so this finds the
    earliest instant at which the task can run uninterrupted.

    Returns (proc_id, start, finish), or None if no processor is ever up
    long enough or the task's input data can't reach any processor.
    """
    if not options:
        return None

    if dynamic_net is None:                       # static: every processor included
        p = min(options, key=lambda q: options[q][0] + options[q][1])
        est, dur = options[p]
        return p, est, est + dur

    instant = -INF                                # -inf = "use each proc's own EST"
    while True:
        best = None
        for proc_id, (est, dur) in options.items():
            start = max(est, instant)
            if not processor_stays_up(dynamic_net, proc_id, start, start + dur):
                continue
            if not output_ok(proc_id, start + dur):
                continue
            if best is None or start + dur < best[2]:
                best = (proc_id, start, start + dur)
        if best is not None:
            return best

        earliest = min(est for est, _ in options.values())
        instant  = dynamic_net.next_snapshot_time(max(instant, earliest))
        if instant == INF:
            return None


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — SCHEDULING
# ═════════════════════════════════════════════════════════════════════════════

def calc_hepft(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork = None,
    is_dynamic:  bool           = True,
) -> dict:
    """
    HEPFT scheduler.

    1. Order tasks by upward rank                    (rank_tasks)
    2. For each task, compute EST/EFT on every processor that exists at any
       time, using integrated communication costs in dynamic mode.
       Processors the input data can't reach (infinite cost) are excluded.
    3. Keep only processors that stay up for the whole run and from which
       the task's output can still reach a processor for each child
       (output_can_leave); pick the smallest EFT. If none qualifies, wait
       for the next snapshot instant and re-check     (choose_processor)
    4. SchedulingError (a RuntimeError) if a task can never be placed
       (no fallback); its .issues maps the task to the reason.

    Returns {task_id: (proc_id, start_time, finish_time)}.
    """
    dynamic = is_dynamic and dynamic_net is not None

    order          = rank_tasks(dag, network, dynamic_net if dynamic else None)
    candidates     = all_processors(network, dynamic_net if dynamic else None)
    schedule       = {}
    proc_available = {p: 0.0 for p in candidates}

    def est_eft(task_id: int, proc_id: int) -> tuple[float, float]:
        task       = dag.nodes[task_id]
        ready_time = 0.0
        for parent_id in task.parents:
            parent_proc, _, parent_eft = schedule[parent_id]
            data_size = dag.edges[(parent_id, task_id)]
            if dynamic:
                comm = dynamic_net.comm_cost_integrated(
                    parent_proc, proc_id, data_size, parent_eft)
            else:
                comm = network.comm_cost(parent_proc, proc_id, data_size)
            ready_time = max(ready_time, parent_eft + comm)
        est = max(ready_time, proc_available[proc_id])
        return est, est + task.comp_costs[proc_id]

    for task_id in order:
        options = {}
        for proc_id in candidates:
            if proc_id not in dag.nodes[task_id].comp_costs:
                continue                          # no cost defined for this processor
            est, eft = est_eft(task_id, proc_id)
            if est != INF:
                options[proc_id] = (est, eft - est)

        if dynamic:
            choice = choose_processor(
                dynamic_net, options,
                lambda p, finish: output_can_leave(
                    dag, dynamic_net, task_id, p, finish, candidates))
        else:
            choice = choose_processor(None, options)
        if choice is None:
            reason = ("its input data can't reach any processor" if not options
                      else "no reachable processor is ever up long enough to run it")
            raise SchedulingError(f"Task {task_id}: {reason}", {task_id: reason})

        best_proc, best_est, best_eft = choice
        schedule[task_id]         = (best_proc, best_est, best_eft)
        proc_available[best_proc] = best_eft

    return schedule