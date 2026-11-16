# hepft_helpers.py
"""
Building blocks for the HEPFT scheduler (hepft.py puts them together).

Every function in Parts 1 and 2 takes an `is_dynamic` flag:
  is_dynamic=True   dynamic HEPFT: plans around the changing network in
                    `dynamic_net` (required)
  is_dynamic=False  static HEPFT, which behaves like HEFT: only needs what
                    HEFT needs (the DAG and the starting network); every
                    processor in the network is always up
`dynamic_net` and `is_dynamic` are always the last two parameters.

  Part 1 — task ranking
      rank_tasks           order tasks by upward rank

  Part 2 — deciding which processors to include
      all_processors       which processors can be used
      processor_stays_up   is a processor up for a task's whole run?
      output_can_leave     can the task's output still reach its children?
      fit_in_gap           earliest idle gap on a processor that fits a task
      earliest_slot        earliest usable start on one processor
      choose_processor     pick the processor that finishes the task first

Processor schedules are INSERTION-BASED, as in canonical HEFT: a task can go
into an idle gap between tasks already placed on a processor, not only after
the last one.

  SchedulingError          raised when a task can never be placed
"""
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


def _require_dynamic_net(dynamic_net: DynamicNetwork | None, func: str):
    """Dynamic mode can't run without a dynamic network."""
    if dynamic_net is None:
        raise ValueError(f"{func}: is_dynamic=True needs a dynamic_net")


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — TASK RANKING
# ═════════════════════════════════════════════════════════════════════════════

def rank_tasks(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork | None = None,
    is_dynamic:  bool                  = True,
) -> list[int]:
    """
    Return task IDs in scheduling order (highest upward rank first).

    Static  (is_dynamic=False): ranks are computed on `network`, as in HEFT.
    Dynamic (is_dynamic=True):  ranks are computed on a time-weighted AVERAGE
             of the snapshots covering [0, estimated makespan], so one unusual
             snapshot can't distort the priority order.
             - Every processor present at some point in the window is kept.
             - Bandwidth is averaged over the whole window, so a link that
               only exists part of the time counts as proportionally slower.
             - Raises ValueError only if the window contains no processors.
    """
    if not is_dynamic or not dag.nodes:
        ranks = dag.compute_ranks(network)
        return sorted(ranks, key=lambda t: ranks[t], reverse=True)

    _require_dynamic_net(dynamic_net, "rank_tasks")

    # Window end = largest static rank + one average task's run time
    static_ranks = dag.compute_ranks(network)
    avg_comp = sum(
        sum(t.comp_costs.values()) / len(t.comp_costs) for t in dag.nodes.values()
    ) / len(dag.nodes)
    t_end = max(static_ranks.values()) + avg_comp

    # Each snapshot lasts from its timestamp until the next one.
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

    # Build the averaged network to rank on
    total    = sum(w for w, _ in segments)
    rank_net = NetworkGraph()
    # Processor objects come from the latest segment that contains each one
    rank_net.processors = {
        p: next(s.processors[p] for _, s in reversed(segments) if p in s.processors)
        for p in procs
    }

    if isinstance(segments[-1][1].bandwidth, (int, float)):
        # Single bandwidth value for the whole network
        rank_net.bandwidth = sum(w * s.bandwidth for w, s in segments) / total
    else:
        # One bandwidth per link; a link missing from a segment adds 0 for it
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

def all_processors(
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork | None = None,
    is_dynamic:  bool                  = True,
) -> set[int]:
    """
    Which processors the scheduler may use.

    Static:  the processors in `network`, as in HEFT.
    Dynamic: every processor that exists at any time: the starting network's
             plus any that join in a later snapshot.
    """
    procs = set(network.processors)
    if not is_dynamic:
        return procs

    _require_dynamic_net(dynamic_net, "all_processors")
    for _, snap in dynamic_net.snapshots:
        procs |= set(snap.processors)
    return procs


def processor_stays_up(
    proc_id:     int,
    start:       float,
    finish:      float,
    dynamic_net: DynamicNetwork | None = None,
    is_dynamic:  bool                  = True,
) -> bool:
    """
    True if proc_id is present for the whole run [start, finish).

    Static:  always True — in a static network processors never go down.
    Dynamic: proc_id must be present at the start and at every snapshot
             change before the finish. Dropping out exactly at the finish
             is fine — the task is already done.
    """
    if not is_dynamic:
        return True

    _require_dynamic_net(dynamic_net, "processor_stays_up")
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
    task_id:     int,
    proc_id:     int,
    finish:      float,
    candidates:  set[int],
    dynamic_net: DynamicNetwork | None = None,
    is_dynamic:  bool                  = True,
) -> bool:
    """
    True if, when task_id finishes on proc_id at `finish`, its output can
    still be sent to its children.

    Static:  always True — HEFT has no such check; links never disappear.
    Dynamic: for each child, the data must be able to reach at least one
             processor that can run that child (has a run time for it and is
             up at some point from `finish` on). Prevents a task's output from
             being stranded on a processor that leaves.
    """
    if not is_dynamic:
        return True

    _require_dynamic_net(dynamic_net, "output_can_leave")

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


def fit_in_gap(
    busy:     list[tuple[float, float]],
    ready:    float,
    duration: float,
) -> float:
    """
    Earliest start >= `ready` at which a task of length `duration` fits on a
    processor without overlapping any of its `busy` intervals.

    busy : the processor's placed tasks as (start, finish), sorted by start

    Checks each idle gap in time order, including the gaps between placed
    tasks, and falls back to after the last task. Same rule in static and
    dynamic mode; reactive.py uses it too.
    """
    start = ready
    for b_start, b_finish in busy:
        if start + duration <= b_start:           # fits in the gap before this task
            return start
        start = max(start, b_finish)              # otherwise try after it
    return start


def earliest_slot(
    proc_id:     int,
    ready:       float,
    duration:    float,
    busy:        list[tuple[float, float]],
    output_ok=   lambda proc_id, finish: True,
    dynamic_net: DynamicNetwork | None = None,
    is_dynamic:  bool                  = True,
) -> float | None:
    """
    Earliest start >= `ready` for a task on proc_id.

    Static:  the earliest idle gap that fits (fit_in_gap), as in canonical HEFT.
    Dynamic: the earliest idle gap that fits AND where the processor stays up
             for the whole run AND output_ok(proc_id, finish) holds. If a gap
             fails those checks, the search moves to the next network change
             and looks for a gap from there. Processors only appear or
             disappear at snapshot times, so no earlier start can work.

    Returns the start time, or None if no slot ever works.
    """
    start = ready
    while True:
        start = fit_in_gap(busy, start, duration)
        if not is_dynamic:
            return start
        if (processor_stays_up(proc_id, start, start + duration, dynamic_net, is_dynamic)
                and output_ok(proc_id, start + duration)):
            return start
        start = dynamic_net.next_snapshot_time(start)   # try again after the next change
        if start == INF:
            return None


def choose_processor(
    options:     dict[int, tuple[float, float]],
    busy:        dict[int, list[tuple[float, float]]],
    output_ok=   lambda proc_id, finish: True,
    dynamic_net: DynamicNetwork | None = None,
    is_dynamic:  bool                  = True,
) -> tuple[int, float, float] | None:
    """
    Pick the processor for one task: the one where it finishes first.

    options   : {proc_id: (data-ready time, run time)} for every processor
                the task's input data can reach
    busy      : {proc_id: sorted (start, finish) intervals already placed}
    output_ok : extra check on (proc_id, finish); hepft passes
                output_can_leave here (dynamic only)

    For each processor, earliest_slot finds the earliest usable idle gap
    (static: any gap that fits; dynamic: a gap where the processor also
    stays up and the output can leave). The processor with the earliest
    finish wins; ties go to the first processor checked.

    Returns (proc_id, start, finish), or None if no processor ever works.
    """
    if is_dynamic:
        _require_dynamic_net(dynamic_net, "choose_processor")

    best = None
    for proc_id, (ready, dur) in options.items():
        start = earliest_slot(proc_id, ready, dur, busy.get(proc_id, []),
                              output_ok, dynamic_net, is_dynamic)
        if start is None:
            continue
        if best is None or start + dur < best[2]:
            best = (proc_id, start, start + dur)
    return best