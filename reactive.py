from __future__ import annotations

from make_dag        import TaskDAG, Task
from make_network    import NetworkGraph, Processor
from dynamic_network import DynamicNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _comm(
    network:     NetworkGraph,
    dynamic_net: DynamicNetwork | None,
    parent_proc: int,
    proc_id:     int,
    data_size:   float,
    parent_eft:  float,
) -> float:
    """
    Communication cost for sending `data_size` from parent_proc to proc_id,
    starting when the parent finishes (parent_eft, absolute time).

    Matches hepft.py: when a DynamicNetwork is available the cost is
    integrated over the changing bandwidth snapshots; otherwise it falls back
    to the static comm_cost of `network`.
    """
    if dynamic_net is not None:
        return dynamic_net.comm_cost_integrated(
            parent_proc, proc_id, data_size, parent_eft
        )
    return network.comm_cost(parent_proc, proc_id, data_size)


# ─────────────────────────────────────────────────────────────────────────────
# HEFT pass in absolute time with integrated comm costs
# ─────────────────────────────────────────────────────────────────────────────

def _heft_integrated(
    dag:           TaskDAG,
    network:       NetworkGraph,
    dynamic_net:   DynamicNetwork | None,
    pre_committed: dict[int, tuple],
    orig_dag:      TaskDAG,
    event_time:    float,
) -> tuple[dict[int, tuple], list[dict]]:
    """
    Runs the HEFT scheduling loop on `dag` using the processors in `network`
    (the snapshot at event_time), and records, for every task, the EFT
    computed for every processor at the moment of decision.

    Unlike plain calc_heft, all times are ABSOLUTE (wall-clock):
      * processors become available at event_time, not 0;
      * parent finish times are real finish times;
    so dynamic_net.comm_cost_integrated() is evaluated over the correct
    window of bandwidth snapshots — the same way calc_hepft does it.
    Because of that, no rebasing step is needed afterwards.

    pre_committed  — tasks already locked in from previous events; used to
                     compute cross-boundary communication costs.
    orig_dag       — the full DAG; needed to look up edges that cross the
                     sub-DAG boundary.

    Returns
    -------
    schedule  : {task_id: (proc_id, start, finish)}  (absolute times)
    decisions : list of dicts, one per task in scheduling order:
                  task_id, eft_per_proc {proc_id: eft}, chosen_proc, chosen_eft
    """
    ranks        = dag.compute_ranks(network)
    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

    schedule:       dict[int, tuple] = {}
    proc_available: dict[int, float] = {p: event_time for p in network.processors}
    decisions:      list[dict]       = []

    def parent_entries(task_id: int):
        """Yield (parent_proc, parent_eft, data_size) for every scheduled parent."""
        # Parents inside this (sub-)DAG
        for parent_id in dag.nodes[task_id].parents:
            if parent_id in schedule:
                parent_proc, _, parent_eft = schedule[parent_id]
                yield parent_proc, parent_eft, dag.edges[(parent_id, task_id)]
        # Parents committed in a prior event (cross-boundary)
        for parent_id in orig_dag.nodes[task_id].parents:
            if parent_id in pre_committed and parent_id not in schedule:
                parent_proc, _, parent_eft = pre_committed[parent_id]
                yield parent_proc, parent_eft, orig_dag.edges[(parent_id, task_id)]

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]

        best_proc: int | None = None
        best_est:  float      = event_time
        best_eft:  float      = float('inf')
        eft_per_proc: dict[int, float] = {}

        for proc_id in network.processors:
            ready_time = event_time
            reachable  = True

            for parent_proc, parent_eft, data_size in parent_entries(task_id):
                comm = _comm(network, dynamic_net,
                             parent_proc, proc_id, data_size, parent_eft)
                if comm == float('inf'):
                    reachable = False        # no path for this data
                    break
                ready_time = max(ready_time, parent_eft + comm)

            if not reachable:
                continue                     # shown as "unavailable" in trace

            est = max(ready_time, proc_available[proc_id])
            eft = est + task.comp_costs[proc_id]
            eft_per_proc[proc_id] = eft

            if eft < best_eft:
                best_eft, best_est, best_proc = eft, est, proc_id

        # Fallback: every processor was unreachable from some parent.
        # Use the snapshot's static cost and treat a missing link as 0,
        # which matches the behaviour of the original _rebase().
        if best_proc is None:
            for proc_id in network.processors:
                ready_time = event_time
                for parent_proc, parent_eft, data_size in parent_entries(task_id):
                    comm = network.comm_cost(parent_proc, proc_id, data_size)
                    if comm == float('inf'):
                        comm = 0.0
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
                eft_per_proc = dec['eft_per_proc']   # {proc_id: absolute eft}
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
            by_proc.setdefault(pid, []).append(f'T{tid}[{s:.1f}-{e:.1f}]')
        for pid in sorted(by_proc.keys()):
            slot = '  '.join(by_proc[pid]) if by_proc[pid] else ''
            log(f'P{pid + 1} : {slot}')

    # ── Final schedule ────────────────────────────────────────────────────
    log()
    log(SEP)
    log('FINAL SCHEDULE')
    log(SEP)
    by_proc = {p: [] for p in sorted(network.processors.keys())}
    for tid, (pid, s, e) in sorted(final.items(), key=lambda x: x[1][1]):
        by_proc.setdefault(pid, []).append(f'T{tid}[{s:.1f}-{e:.1f}]')
    for pid in sorted(by_proc.keys()):
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
    Event-driven reactive scheduler.  Re-runs HEFT from scratch on the
    remaining tasks at every processor topology change.

    Communication costs use dynamic_net.comm_cost_integrated(), evaluated
    from each parent's actual finish time — the same cost model as
    calc_hepft in dynamic mode — so the two schedulers are compared on
    equal terms.  Scheduling is done directly in absolute time, so the
    old zero-based calc_heft + _rebase step is no longer needed.

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

    # ── Initial plan (t = 0, base network, integrated comm costs) ─────────
    event_log: list[dict] = []

    current_plan, dec_init = _heft_integrated(
        dag, network, dynamic_net, {}, dag, event_time=0.0
    )

    if do_trace:
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

        replanned, decisions = _heft_integrated(
            sub, snapshot, dynamic_net, actual, dag, event_time=event_time
        )
        current_plan = {**actual, **replanned}

        if do_trace:
            event_log.append({
                'step':             event_idx,
                'event_time':       event_time,
                'decisions':        decisions,
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