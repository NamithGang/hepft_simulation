from __future__ import annotations

from make_dag        import TaskDAG, Task
from make_network    import NetworkGraph, Processor
from dynamic_network import DynamicNetwork
from heft            import calc_heft


# ─────────────────────────────────────────────────────────────────────────────
# Original helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

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
    raw:        dict[int, tuple],
    sub:        TaskDAG,
    orig_dag:   TaskDAG,
    fixed:      dict[int, tuple],
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
    topo    = list(reversed(sub._topological_sort()))  # roots first
    rebased: dict[int, tuple] = {}

    for tid in topo:
        proc_id, raw_start, raw_finish = raw[tid]
        duration = raw_finish - raw_start
        floor    = event_time

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
                floor = max(floor, rebased[parent_id][2])

        new_start    = max(raw_start, floor)
        rebased[tid] = (proc_id, new_start, new_start + duration)

    return rebased


# ─────────────────────────────────────────────────────────────────────────────
# HEFT re-run that captures per-decision EFT values for the trace
# ─────────────────────────────────────────────────────────────────────────────

def _heft_instrumented(
    dag:           TaskDAG,
    network:       NetworkGraph,
    pre_committed: dict[int, tuple],
    orig_dag:      TaskDAG,
) -> tuple[dict[int, tuple], list[dict]]:
    """
    Runs the HEFT scheduling loop and records, for every task, the exact EFT
    computed for every processor at the moment of decision.

    pre_committed  — tasks already locked in from previous events; used to
                     compute cross-boundary communication costs correctly.
    orig_dag       — the full DAG; needed to look up edges that cross the
                     sub-DAG boundary.

    Returns
    -------
    schedule  : {task_id: (proc_id, start, finish)}  (zero-based times)
    decisions : list of dicts, one per task in scheduling order:
                  task_id, eft_per_proc {proc_id: eft}, chosen_proc, chosen_eft
    """
    ranks        = dag.compute_ranks(network)
    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

    schedule:      dict[int, tuple] = {}
    proc_available = {p: 0.0 for p in network.processors}
    decisions:     list[dict] = []

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]

        best_proc: int | None = None
        best_est:  float      = 0.0
        best_eft:  float      = float('inf')
        eft_per_proc: dict[int, float] = {}

        for proc_id in network.processors:
            ready_time = 0.0

            # Parents inside this (sub-)DAG
            for parent_id in task.parents:
                if parent_id in schedule:
                    parent_proc, _, parent_eft = schedule[parent_id]
                    data_size  = dag.edges[(parent_id, task_id)]
                    comm       = network.comm_cost(parent_proc, proc_id, data_size)
                    ready_time = max(ready_time, parent_eft + comm)

            # Parents committed in a prior event (cross-boundary)
            orig_parents = orig_dag.nodes[task_id].parents if orig_dag is not None else []
            for parent_id in orig_parents:
                if parent_id in pre_committed and parent_id not in schedule:
                    parent_proc, _, parent_eft = pre_committed[parent_id]
                    data_size  = orig_dag.edges[(parent_id, task_id)]
                    comm       = network.comm_cost(parent_proc, proc_id, data_size)
                    ready_time = max(ready_time, parent_eft + comm)

            est = max(ready_time, proc_available[proc_id])
            eft = est + task.comp_costs[proc_id]
            eft_per_proc[proc_id] = eft

            if eft < best_eft:
                best_eft, best_est, best_proc = eft, est, proc_id

        schedule[task_id]         = (best_proc, best_est, best_eft)
        proc_available[best_proc] = best_eft

        decisions.append({
            'task_id':      task_id,
            'eft_per_proc': eft_per_proc,
            'chosen_proc':  best_proc,
            'chosen_eft':   best_eft,
        })

    return schedule, decisions


# ─────────────────────────────────────────────────────────────────────────────
# Trace formatter
# ─────────────────────────────────────────────────────────────────────────────

SEP = '=' * 50

def _write_trace(
    dag:         TaskDAG,
    network:     NetworkGraph,
    final:       dict[int, tuple],
    event_log:   list[dict],
    trace_file:  str,
) -> None:
    lines: list[str] = []

    def log(msg: str = ''):
        lines.append(msg)

    # ── Task priorities ───────────────────────────────────────────────────
    ranks    = dag.compute_ranks(network)
    priority = sorted(dag.nodes.keys(), key=lambda t: ranks[t], reverse=True)

    log(SEP)
    log('TASK PRIORITIES')
    log(SEP)
    log(f"{'Task':<10} {'Upward Rank':>12}")
    log('-' * 22)
    for tid in priority:
        log(f"T{tid:<9} {ranks[tid]:>12.2f}")
    log()
    log('Scheduling Order:')
    log(' -> '.join(f'T{t}' for t in priority))

    # ── One block per scheduling event ────────────────────────────────────
    for ev in event_log:
        step_num  = ev['step']
        ev_time   = ev['event_time']
        decisions = ev['decisions']          # list[dict] — one per task
        committed = ev['committed_so_far']   # full plan after this event
        locked    = ev['locked']             # tasks committed before this event
        active    = ev['active_procs']       # processor IDs in this snapshot
        lost      = ev['lost_procs']
        gained    = ev['gained_procs']

        log()
        log(SEP)
        label = 'INITIAL PLAN' if step_num == 0 else f'RESCHEDULE at t={ev_time:.2f}'
        log(f'STEP {step_num}  [{label}]')
        log(SEP)

        # Context for reschedule steps
        if step_num > 0:
            log(f'Active processors : {sorted(active)}')
            if lost:
                log(f'Lost              : {sorted(lost)}')
            if gained:
                log(f'Gained            : {sorted(gained)}')
            if locked:
                log(f'Locked tasks      : {[f"T{t}" for t in sorted(locked)]}')
            log()

        if not decisions:
            log('  (no tasks to schedule)')
        else:
            for dec in decisions:
                task_id      = dec['task_id']
                eft_per_proc = dec['eft_per_proc']   # {proc_id: rebased_eft}
                chosen_proc  = dec['chosen_proc']
                chosen_eft   = dec['chosen_eft']

                log(f'Current Task: T{task_id}')
                log()
                log('Processor Evaluation')
                log('-' * 20)
                for pid in sorted(network.processors.keys()):
                    plabel = f'P{pid + 1}'
                    if pid not in eft_per_proc:
                        log(f'{plabel} : EFT = -- (unavailable)')
                    else:
                        marker = '  *' if pid == chosen_proc else ''
                        log(f'{plabel} : EFT = {eft_per_proc[pid]:.2f}{marker}')
                log()
                log(f'Selected Processor: P{chosen_proc + 1}')
                log(f'Reason: Minimum EFT = {chosen_eft:.2f}')
                log()

        # Running schedule snapshot
        log('Current Schedule')
        log('-' * 16)
        by_proc: dict[int, list[str]] = {p: [] for p in sorted(network.processors.keys())}
        for tid, (pid, s, e) in sorted(committed.items(), key=lambda x: x[1][1]):
            by_proc[pid].append(f'T{tid}[{s:.1f}-{e:.1f}]')
        for pid in sorted(network.processors.keys()):
            slot = '  '.join(by_proc[pid]) if by_proc[pid] else ''
            log(f'P{pid + 1} : {slot}')

    # ── Final schedule ────────────────────────────────────────────────────
    log()
    log(SEP)
    log('FINAL SCHEDULE')
    log(SEP)
    by_proc = {p: [] for p in sorted(network.processors.keys())}
    for tid, (pid, s, e) in sorted(final.items(), key=lambda x: x[1][1]):
        by_proc[pid].append(f'T{tid}[{s:.1f}-{e:.1f}]')
    for pid in sorted(network.processors.keys()):
        slot = '  '.join(by_proc[pid]) if by_proc[pid] else '(idle)'
        log(f'P{pid + 1} : {slot}')

    if final:
        ms = max(v[2] for v in final.values()) - min(v[1] for v in final.values())
        log()
        log(f'Makespan: {ms:.4f}')

    with open(trace_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def simulate_reactive(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork,
    trace_file:  str | None = None,
) -> dict[int, tuple]:
    """
    Event-driven reactive scheduler.  Calls calc_heft() from scratch
    on remaining tasks at every processor topology change.

    If trace_file is given, writes a step-by-step HEFT-style trace showing
    the upward-rank order, per-processor EFT at the moment of every decision,
    the chosen processor and reason, and a running schedule snapshot.

    Returns: {task_id: (proc_id, actual_start, actual_finish)}
    """
    do_trace = trace_file is not None

    # ── Topology-change events ────────────────────────────────────────────
    topology_events: list[float] = []
    prev_procs: set | None = None
    for ts, net in sorted(dynamic_net.snapshots, key=lambda x: x[0]):
        curr_procs = set(net.processors.keys())
        if prev_procs is None or curr_procs != prev_procs:
            topology_events.append(ts)
            prev_procs = curr_procs

    # ── Initial plan ──────────────────────────────────────────────────────
    event_log: list[dict] = []

    if do_trace:
        raw_init, dec_init = _heft_instrumented(dag, network, {}, dag)
        current_plan: dict[int, tuple] = raw_init
        # Initial plan: EFTs are already wall-clock (start at 0), no rebase needed
        event_log.append({
            'step':             0,
            'event_time':       0.0,
            'decisions':        dec_init,
            'committed_so_far': dict(current_plan),
            'locked':           set(),
            'active_procs':     set(network.processors.keys()),
            'lost_procs':       set(),
            'gained_procs':     set(),
        })
    else:
        current_plan = calc_heft(dag, network)

    actual: dict[int, tuple] = {}
    prev_proc_set = set(network.processors.keys())

    # ── Event loop ────────────────────────────────────────────────────────
    for event_idx, event_time in enumerate(topology_events, start=1):
        snapshot      = dynamic_net.pred_net_func(event_time)
        curr_proc_set = set(snapshot.processors.keys())

        # Commit tasks whose planned start ≤ event_time
        for task_id, (proc_id, start, finish) in list(current_plan.items()):
            if start <= event_time and task_id not in actual:
                actual[task_id] = (proc_id, start, finish)

        remaining = set(dag.nodes) - set(actual)
        if not remaining or not snapshot.processors:
            prev_proc_set = curr_proc_set
            continue

        sub = _sub_dag(dag, remaining)

        if do_trace:
            raw, decisions = _heft_instrumented(sub, snapshot, actual, dag)
        else:
            raw       = calc_heft(sub, snapshot)
            decisions = []

        shifted      = _rebase(raw, sub, dag, actual, snapshot, event_time)
        current_plan = {**actual, **shifted}

        if do_trace:
            # Rebase each decision's EFTs by the same shift applied to that task.
            # All processors in a task's evaluation shift by the same delta because
            # the rebase floor (event_time + comm from committed parents) is
            # proc-independent for the winning processor; we apply the winner's
            # shift uniformly so the table stays internally consistent.
            rebased_decisions = []
            for dec in decisions:
                tid = dec['task_id']
                if tid in shifted:
                    _, _, rebased_finish = shifted[tid]
                    _, _, raw_finish     = raw[tid]
                    shift = rebased_finish - raw_finish
                    rebased_decisions.append({
                        'task_id':      tid,
                        'eft_per_proc': {pid: e + shift for pid, e in dec['eft_per_proc'].items()},
                        'chosen_proc':  dec['chosen_proc'],
                        'chosen_eft':   rebased_finish,
                    })
                else:
                    rebased_decisions.append(dec)

            event_log.append({
                'step':             event_idx,
                'event_time':       event_time,
                'decisions':        rebased_decisions,
                'committed_so_far': dict(current_plan),
                'locked':           set(actual.keys()),
                'active_procs':     curr_proc_set,
                'lost_procs':       prev_proc_set - curr_proc_set,
                'gained_procs':     curr_proc_set - prev_proc_set,
            })

        prev_proc_set = curr_proc_set

    # ── Commit anything left after the last event ─────────────────────────
    for task_id, entry in current_plan.items():
        if task_id not in actual:
            actual[task_id] = entry

    # ── Write trace ───────────────────────────────────────────────────────
    if do_trace:
        _write_trace(dag, network, actual, event_log, trace_file)

    return actual