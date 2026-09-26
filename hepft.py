from __future__ import annotations

from make_dag      import TaskDAG, Task
from make_network  import NetworkGraph, Processor
from demo          import create_dag, create_network, create_dynamic_network
from dynamic_network import DynamicNetwork


# ── private helpers ──────────────────────────────────────────────────────────

def _avg_network(
    base_network: NetworkGraph,
    dynamic_net:  DynamicNetwork,
    t_start:      float,
    t_end:        float,
) -> NetworkGraph:
    """
    Return a NetworkGraph whose bandwidth equals the time-weighted average of
    all snapshots that overlap [t_start, t_end].

    Only processors present in *every* overlapping snapshot are kept — a
    processor that disappears for any portion of the window is excluded,
    mirroring the window check in the dynamic selection loop.

    If no snapshots fall in the window (or t_end <= t_start) we fall back to
    the base network unchanged.
    """
    if t_end <= t_start:
        return base_network

    # Collect (snapshot_network, weight) pairs for snapshots active inside
    # [t_start, t_end].  We treat each snapshot as active from its timestamp
    # up to the next snapshot's timestamp (or t_end, whichever comes first).
    sorted_snaps = sorted(dynamic_net.snapshots, key=lambda x: x[0])

    # Find the snapshot active at t_start (last one with ts <= t_start)
    active_idx = 0
    for i, (ts, _) in enumerate(sorted_snaps):
        if ts <= t_start:
            active_idx = i

    weighted_bw:   dict[tuple, float] = {}
    total_weight:  float              = 0.0
    proc_presence: dict               = {}          # proc_id -> total weighted presence
    proc_possible: set                = None        # intersection of all seen procs

    cursor = t_start
    idx    = active_idx

    while cursor < t_end and idx < len(sorted_snaps):
        snap_ts,  snap_net = sorted_snaps[idx]
        next_ts = sorted_snaps[idx + 1][0] if idx + 1 < len(sorted_snaps) else float('inf')

        seg_start  = max(cursor, snap_ts)
        seg_end    = min(t_end,  next_ts)
        weight     = seg_end - seg_start

        if weight <= 0:
            idx += 1
            cursor = seg_end
            continue

        total_weight += weight

        # Track which processors are present in this segment
        seg_procs = set(snap_net.processors.keys())
        if proc_possible is None:
            proc_possible = set(seg_procs)
        else:
            proc_possible &= seg_procs        # intersection — must survive all segments

        for proc_id in seg_procs:
            proc_presence[proc_id] = proc_presence.get(proc_id, 0.0) + weight

        # Accumulate weighted bandwidth for each processor-pair present
        for (p1, p2), bw in snap_net.bandwidth.items():
            key = (p1, p2)
            weighted_bw[key] = weighted_bw.get(key, 0.0) + bw * weight

        cursor = seg_end
        idx   += 1

    if total_weight == 0 or proc_possible is None:
        return base_network

    # Build average bandwidth dict (only for pairs where both procs survive)
    avg_bw: dict[tuple, float] = {}
    for (p1, p2), wb in weighted_bw.items():
        if p1 in proc_possible and p2 in proc_possible:
            avg_bw[(p1, p2)] = wb / total_weight

    # Construct a lightweight NetworkGraph clone with averaged values.
    # NetworkGraph stores processors in a plain dict at .processors —
    # there is no add_processor() method, so we assign the dict directly.
    avg_net = NetworkGraph()

    # Populate .processors: prefer the object from base_network, fall back to
    # the last snapshot that had that processor.
    avg_net.processors = {}
    for proc_id in proc_possible:
        proc = (base_network.processors.get(proc_id)
                or next(
                    (sn.processors[proc_id]
                     for _, sn in reversed(sorted_snaps)
                     if proc_id in sn.processors),
                    None
                ))
        if proc is not None:
            avg_net.processors[proc_id] = proc

    # .bandwidth holds either a scalar (uniform) or a per-pair dict.
    # If base_network uses a scalar, set a scalar on avg_net too.
    if isinstance(getattr(base_network, 'bandwidth', None), (int, float)):
        # Compute a time-weighted average of the scalar bandwidth values
        # seen across all snapshots in the window.
        scalar_total  = 0.0
        scalar_weight = 0.0
        cursor2 = t_start
        idx2    = active_idx
        while cursor2 < t_end and idx2 < len(sorted_snaps):
            snap_ts2, snap_net2 = sorted_snaps[idx2]
            next_ts2 = sorted_snaps[idx2 + 1][0] if idx2 + 1 < len(sorted_snaps) else float('inf')
            seg_s2 = max(cursor2, snap_ts2)
            seg_e2 = min(t_end, next_ts2)
            w2     = seg_e2 - seg_s2
            if w2 > 0 and isinstance(getattr(snap_net2, 'bandwidth', None), (int, float)):
                scalar_total  += snap_net2.bandwidth * w2
                scalar_weight += w2
            cursor2 = seg_e2
            idx2   += 1
        avg_net.bandwidth = (scalar_total / scalar_weight
                             if scalar_weight > 0
                             else base_network.bandwidth)
    else:
        # Per-pair bandwidth dict — already computed above.
        avg_net.bandwidth = avg_bw

    return avg_net


def _compute_est(
    task_id:      int,
    proc_id:      int,
    dag:          TaskDAG,
    schedule:     dict,
    proc_available: dict,
    network:      NetworkGraph,
    dynamic_net:  DynamicNetwork,
    is_dynamic:   bool,
) -> float:
    """
    Compute the Earliest Start Time for task_id on proc_id.

    Static path:  uses plain comm_cost on the fixed network.
    Dynamic path: integrates over changing bandwidth snapshots.
    """
    task       = dag.nodes[task_id]
    ready_time = 0.0

    for parent_id in task.parents:
        parent_proc, _, parent_eft = schedule[parent_id]
        data_size = dag.edges[(parent_id, task_id)]

        if is_dynamic:
            comm = dynamic_net.pred_net_func(parent_eft).comm_cost_integrated(
                parent_proc, proc_id, data_size, parent_eft, dynamic_net,
                fallback_bandwidth=network.bandwidth
            )
        else:
            comm = network.comm_cost(
                parent_proc, proc_id, data_size,
                fallback_bandwidth=None
            )

        ready_time = max(ready_time, parent_eft + comm)

    return max(ready_time, proc_available[proc_id])


def _fallback_assign(
    task_id:      int,
    dag:          TaskDAG,
    schedule:     dict,
    proc_available: dict,
    network:      NetworkGraph,
) -> tuple:
    """
    Fallback processor selection using static network comm costs.
    Called when every processor fails the window check in dynamic mode.

    Returns (best_proc, best_est, best_eft).
    """
    task = dag.nodes[task_id]
    best_proc, best_est, best_eft = None, None, float('inf')

    for proc_id in network.processors:
        ready_time = 0.0

        for parent_id in task.parents:
            parent_proc, _, parent_eft = schedule[parent_id]
            data_size = dag.edges[(parent_id, task_id)]
            comm      = network.comm_cost(parent_proc, proc_id, data_size)
            ready_time = max(ready_time, parent_eft + comm)

        est = max(ready_time, proc_available[proc_id])
        eft = est + task.comp_costs[proc_id]

        if eft < best_eft:
            best_eft, best_est, best_proc = eft, est, proc_id

    return best_proc, best_est, best_eft


def _child_cone(dag: TaskDAG, seed_ids: set[int]) -> set[int]:
    """
    Return the set of all descendants of seed_ids (inclusive) in dag,
    found via BFS on children edges.
    """
    cone  = set(seed_ids)
    queue = list(seed_ids)
    while queue:
        tid = queue.pop()
        for child_id in dag.nodes[tid].children:
            if child_id not in cone:
                cone.add(child_id)
                queue.append(child_id)
    return cone


def _build_sub_dag(dag: TaskDAG, task_ids: set[int]) -> TaskDAG:
    """Build a TaskDAG containing only task_ids with their internal edges."""
    sub = TaskDAG()
    for tid in task_ids:
        sub.nodes[tid] = Task(tid, dict(dag.nodes[tid].comp_costs))
    for tid in task_ids:
        for child_id in dag.nodes[tid].children:
            if child_id in task_ids:
                sub.add_edge(tid, child_id, comm_cost=dag.edges[(tid, child_id)])
    return sub


def _rebase_cone(
    raw:        dict[int, tuple],
    sub:        TaskDAG,
    orig_dag:   TaskDAG,
    fixed:      dict[int, tuple],
    snapshot:   NetworkGraph,
    floor_time: float,
) -> dict[int, tuple]:
    """
    Shift the zero-based times from a sub-DAG schedule so that:
      1. Nothing starts before floor_time.
      2. Tasks whose parent is in fixed start after parent_finish + comm_cost.
      3. Tasks whose parent is also in sub start after that parent's rebased finish.
    """
    topo    = list(reversed(sub._topological_sort()))   # roots first
    rebased: dict[int, tuple] = {}

    for tid in topo:
        proc_id, raw_start, raw_finish = raw[tid]
        duration = raw_finish - raw_start
        floor    = floor_time

        # Cross-boundary parents (committed outside the cone)
        for parent_id in orig_dag.nodes[tid].parents:
            if parent_id in fixed:
                parent_proc, _, parent_finish = fixed[parent_id]
                data_size = orig_dag.edges[(parent_id, tid)]
                comm = snapshot.comm_cost(
                    parent_proc, proc_id, data_size, fallback_bandwidth=None
                )
                if comm == float('inf'):
                    comm = 0.0
                floor = max(floor, parent_finish + comm)

        # Intra-cone parents
        for parent_id in sub.nodes[tid].parents:
            if parent_id in rebased:
                floor = max(floor, rebased[parent_id][2])

        new_start       = max(raw_start, floor)
        rebased[tid]    = (proc_id, new_start, new_start + duration)

    return rebased


def _mini_reactive(
    dag:         TaskDAG,
    base_network: NetworkGraph,
    dynamic_net: DynamicNetwork,
    schedule:    dict[int, tuple],
) -> dict[int, tuple]:
    """
    Fix 2 — mini-reactive replanning.

    Scans the current schedule for any task whose assigned processor is absent
    from the dynamic network at ANY point during [est, eft] (i.e., it is
    stranded, or its EST is now stale because a parent was moved).  For each
    stranded task we collect its child-cone, drop those entries from the
    schedule, rebuild a sub-DAG, and re-run calc_hepft on it using a static
    snapshot taken at the task's planned EST.  The resulting times are rebased
    and spliced back in.

    The scan repeats until no more stranded tasks are found (rare but possible
    when a re-planned task itself starts on a proc that later fails).  In
    practice this converges in 1-2 passes.

    Returns an updated schedule dict.
    """
    MAX_PASSES = 5          # guard against degenerate cycles

    for _ in range(MAX_PASSES):
        stranded_seeds: set[int] = set()

        for task_id, (proc_id, est, eft) in schedule.items():
            # Quick boundary checks first
            if not dynamic_net.pred_net_func(est).has_processor(proc_id):
                stranded_seeds.add(task_id)
                continue
            if not dynamic_net.pred_net_func(eft).has_processor(proc_id):
                stranded_seeds.add(task_id)
                continue
            # Walk interior snapshots
            t_cursor = est
            failed   = False
            while True:
                next_t = dynamic_net.next_snapshot_time(t_cursor)
                if next_t == float('inf') or next_t >= eft:
                    break
                if not dynamic_net.pred_net_func(next_t).has_processor(proc_id):
                    failed = True
                    break
                t_cursor = next_t
            if failed:
                stranded_seeds.add(task_id)

        if not stranded_seeds:
            break   # nothing left to fix

        # Expand each stranded seed to its full child-cone
        cone = _child_cone(dag, stranded_seeds)

        # Everything outside the cone is "fixed" (committed)
        fixed = {tid: entry for tid, entry in schedule.items() if tid not in cone}

        # Build sub-DAG and pick the snapshot at the earliest stranded EST
        floor_time = min(schedule[tid][1] for tid in stranded_seeds)
        snapshot   = dynamic_net.pred_net_func(floor_time)

        # Only keep processors that are actually present in this snapshot
        if not snapshot.processors:
            snapshot = base_network       # ultimate fallback

        sub = _build_sub_dag(dag, cone)

        # Re-schedule the cone on the snapshot (static, no window checks,
        # no integrated costs — we want plain HEFT quality on a fixed graph)
        raw = calc_hepft(
            sub, snapshot,
            dynamic_net=None,
            is_dynamic=False,
        )

        # Rebase relative to committed tasks and floor_time
        rebased = _rebase_cone(raw, sub, dag, fixed, snapshot, floor_time)

        schedule = {**fixed, **rebased}

    return schedule


# ── public entry point ───────────────────────────────────────────────────────

def calc_hepft(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork = None,
    is_dynamic:  bool           = True,
) -> dict:
    """
    HEPFT scheduler with two gap-closing enhancements:

    Fix 1 — Windowed rank averaging
        Instead of computing upward ranks against a single predicted snapshot,
        we build a time-weighted average NetworkGraph covering each task's
        likely execution window.  This prevents single-point mispredictions
        from distorting the priority order.

    Fix 2 — Mini-reactive replanning  (dynamic mode only)
        After the main forward pass, any task whose assigned processor is
        absent from the dynamic network during [est, eft] — plus its entire
        child-cone — is rescheduled using a static HEPFT call on the snapshot
        at the stranded task's planned start time.  This closes most of the
        gap to a fully reactive scheduler without the cost of full replanning.
    """

    # ── Fix 1: build an averaged network for rank computation ─────────────
    if is_dynamic and dynamic_net is not None:
        # Estimate the overall makespan window using the static network as a
        # proxy so we have bounds before any tasks are scheduled.
        static_ranks  = dag.compute_ranks(network)
        avg_comp      = (
            sum(
                sum(dag.nodes[tid].comp_costs.values()) / len(dag.nodes[tid].comp_costs)
                for tid in dag.nodes
            ) / len(dag.nodes)
            if dag.nodes else 1.0
        )
        # Window: t=0 to estimated makespan (max rank + one task's avg cost)
        t_window_end  = max(static_ranks.values()) + avg_comp if static_ranks else avg_comp
        rank_network  = _avg_network(network, dynamic_net, t_start=0.0, t_end=t_window_end)
        ranks         = dag.compute_ranks(rank_network)
    elif is_dynamic:
        ranks = dag.compute_ranks(network, dynamic_network=dynamic_net)
    else:
        ranks = dag.compute_ranks(network)

    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

    schedule       = {}
    proc_available = {proc_id: 0.0 for proc_id in network.processors}

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]
        best_proc, best_est, best_eft = None, None, float('inf')

        if is_dynamic:
            # ── dynamic processor selection ────────────────────────────────
            # uses integrated comm costs and window checks against snapshots
            for proc_id in network.processors:

                est = _compute_est(
                    task_id, proc_id, dag, schedule, proc_available,
                    network, dynamic_net, is_dynamic=True
                )
                eft = est + task.comp_costs[proc_id]

                # Window check: skip if proc goes down during [est, eft]
                proc_goes_down = False

                if not dynamic_net.pred_net_func(est).has_processor(proc_id):
                    proc_goes_down = True
                elif not dynamic_net.pred_net_func(eft).has_processor(proc_id):
                    proc_goes_down = True
                else:
                    t_cursor = est
                    while True:
                        next_t = dynamic_net.next_snapshot_time(t_cursor)
                        if next_t == float('inf') or next_t >= eft:
                            break
                        if not dynamic_net.pred_net_func(next_t).has_processor(proc_id):
                            proc_goes_down = True
                            break
                        t_cursor = next_t

                if proc_goes_down:
                    continue

                if eft < best_eft:
                    best_eft, best_est, best_proc = eft, est, proc_id

        else:
            # ── static processor selection ─────────────────────────────────
            for proc_id in network.processors:

                est = _compute_est(
                    task_id, proc_id, dag, schedule, proc_available,
                    network, dynamic_net, is_dynamic=False
                )
                eft = est + task.comp_costs[proc_id]

                if eft < best_eft:
                    best_eft, best_est, best_proc = eft, est, proc_id

        # ── fallback ──────────────────────────────────────────────────────
        if best_proc is None:
            best_proc, best_est, best_eft = _fallback_assign(
                task_id, dag, schedule, proc_available, network
            )

        schedule[task_id]         = (best_proc, best_est, best_eft)
        proc_available[best_proc] = best_eft

    # ── Fix 2: mini-reactive replanning pass ──────────────────────────────
    if is_dynamic and dynamic_net is not None:
        schedule = _mini_reactive(dag, network, dynamic_net, schedule)

    return schedule