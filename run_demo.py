from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from demo import create_dag, create_network, create_dynamic_network
from dynamic_network import DynamicNetwork
from heft import calc_heft
from hepft import calc_hepft
from reactive import simulate_reactive
from oracle_ilp import calc_oracle_ilp

import traceback

INF = float('inf')


# ── issue log ───────────────────────────────────────────────────────────────
#
# Anything that goes wrong (an algorithm raising an error, or tasks failing
# when a schedule is replayed) is written to ISSUE_LOG as one line per task:
#     <test case> | <algorithm> | <task> | <issue>
# and the run carries on with the remaining algorithms and test cases.

ISSUE_LOG = "run_issues.txt"


class IssueLog:
    def __init__(self, path: str = ISSUE_LOG):
        self.path  = path
        self.count = 0
        with open(self.path, 'w') as f:
            f.write("test case | algorithm | task | issue\n")
            f.write("-" * 80 + "\n")

    def add(self, case: str, algorithm: str, task: str, issue: str):
        self.count += 1
        with open(self.path, 'a') as f:        # append, so a crash keeps earlier lines
            f.write(f"{case} | {algorithm} | {task} | {issue}\n")


def _plan(log: IssueLog, case: str, algorithm: str, fn, *args):
    """
    Run a scheduler. If it raises, log the problem (one line per task when the
    error carries per-task details) and return None so the run can continue.
    """
    try:
        return fn(*args)
    except Exception as e:
        issues = getattr(e, 'issues', None)
        if issues:
            for tid, why in sorted(issues.items()):
                log.add(case, algorithm, f"T{tid}", why)
        else:
            log.add(case, algorithm, "-", f"{type(e).__name__}: {e}")
        print(f"  ! {algorithm} failed: {e}  (details in {log.path})")
        return None


def _log_replay_failures(log: IssueLog, case: str, algorithm: str,
                         reasons: dict[int, str]):
    """Log tasks that failed during the replay; root causes first."""
    roots = {t: r for t, r in reasons.items() if not r.startswith("parent ")}
    deps  = {t: r for t, r in reasons.items() if r.startswith("parent ")}
    for tid, why in sorted(roots.items()) + sorted(deps.items()):
        log.add(case, f"{algorithm} (replay)", f"T{tid}", why)
    if reasons:
        print(f"  ! {algorithm}: {len(reasons)} task(s) failed in the replay "
              f"({len(roots)} on their own)  (details in {log.path})")


# ── simulation helpers ──────────────────────────────────────────────────────
#
# simulate_on_dynamic uses the SAME rules as simulate_reactive, so every
# algorithm is scored the same way:
#   * communication cost = dynamic_net.comm_cost_integrated(), measured from
#     the parent's actual finish time
#   * a processor must be up for a task's whole run [start, finish)
#       - down when the task is ready  -> the task waits until it is back
#       - drops partway through        -> work is lost, restart from scratch
#         when it is back
#   * a task that can never run (processor never returns, or its input data
#     can never reach it) FAILS, and so do its descendants; a failed task is
#     recorded as (proc, inf, inf), which makes the makespan inf
#
# The one difference is what a static schedule cannot do: it keeps every
# task on its planned processor, in planned order. simulate_reactive can move
# killed tasks and re-run lost outputs — that is its advantage.

def _proc_up(dynamic_net, proc_id, t):
    return dynamic_net.pred_net_func(t).has_processor(proc_id)


def _next_up(dynamic_net, proc_id, t):
    """Earliest time >= t at which proc_id is present (inf if never)."""
    while not _proc_up(dynamic_net, proc_id, t):
        t = dynamic_net.next_snapshot_time(t)
        if t == INF:
            return INF
    return t


def _first_drop(dynamic_net, proc_id, start, finish):
    """First time in (start, finish) at which proc_id disappears, else None."""
    t = start
    while True:
        t = dynamic_net.next_snapshot_time(t)
        if t == INF or t >= finish:
            return None
        if not _proc_up(dynamic_net, proc_id, t):
            return t


def simulate_on_dynamic(dag, dynamic_net, schedule, reasons=None):
    """
    Re-play a *planned* schedule against the true dynamic network and return
    the actual start/finish times.

    schedule: {task_id: (proc_id, planned_start, planned_finish)}
    returns : {task_id: (proc_id, actual_start, actual_finish)}
              failed tasks are (proc_id, inf, inf)
    reasons : optional dict, filled with {task_id: why it failed}
    """
    if reasons is None:
        reasons = {}
    if not schedule:
        return {}
    ordered = sorted(schedule.items(), key=lambda item: item[1][1])

    actual         = {}
    proc_available = {}

    for task_id, (proc_id, _, _) in ordered:
        ready_time = proc_available.get(proc_id, 0.0)

        for parent_id in dag.nodes[task_id].parents:
            parent_proc, _, parent_finish = actual[parent_id]
            if parent_finish == INF:                  # parent failed
                ready_time = INF
                reasons[task_id] = f"parent T{parent_id} failed"
                break
            data_size = dag.edges[(parent_id, task_id)]
            comm = dynamic_net.comm_cost_integrated(
                parent_proc, proc_id, data_size, parent_finish)
            if comm == INF:
                reasons[task_id] = (f"data from T{parent_id} (on P{parent_proc + 1}, "
                                    f"finished t={parent_finish:.2f}) can never reach "
                                    f"P{proc_id + 1}")
                ready_time = INF
                break
            ready_time = max(ready_time, parent_finish + comm)

        duration = dag.nodes[task_id].comp_costs[proc_id]
        start    = ready_time
        while start != INF:
            start = _next_up(dynamic_net, proc_id, start)   # wait if down
            if start == INF:
                break
            drop = _first_drop(dynamic_net, proc_id, start, start + duration)
            if drop is None:
                break                                         # ran to completion
            start = drop                                      # lost work, retry

        if start == INF:
            reasons.setdefault(task_id, f"P{proc_id + 1} is never up long enough "
                                        f"to run it (needs {duration:.2f})")
            actual[task_id] = (proc_id, INF, INF)
            continue

        actual[task_id]         = (proc_id, start, start + duration)
        proc_available[proc_id] = start + duration

    return actual


def _makespan(schedule: dict) -> float:
    if not schedule:
        return float('inf')
    return (max(v[2] for v in schedule.values())
            - min(v[1] for v in schedule.values()))


# ── auxiliary metrics ───────────────────────────────────────────────────────

def _cp_min(dag: TaskDAG) -> float:
    from collections import deque
    in_degree = {tid: len(t.parents) for tid, t in dag.nodes.items()}
    children  = {tid: [] for tid in dag.nodes}
    for (pid, cid) in dag.edges:
        children[pid].append(cid)
    dist  = {}
    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    while queue:
        tid      = queue.popleft()
        min_cost = min(dag.nodes[tid].comp_costs.values())
        dist[tid] = max((dist[p] for p in dag.nodes[tid].parents), default=0.0) + min_cost
        for cid in children[tid]:
            in_degree[cid] -= 1
            if in_degree[cid] == 0:
                queue.append(cid)
    return max(dist.values(), default=0.0)


def _sequential_time(dag: TaskDAG) -> float:
    return sum(min(t.comp_costs.values()) for t in dag.nodes.values())


def _mean_uptime(dynamic_net: DynamicNetwork, network: NetworkGraph) -> float:
    """
    Mean of dynamic_net.proc_volatility() over the processors: the average
    length of an up-period. LOWER = MORE VOLATILE (processors flicker more).
    """
    vals = [dynamic_net.proc_volatility(pid) for pid in network.processors]
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(v: float, width: int = 10) -> str:
    """Format a makespan; failed runs (inf) print as FAILED."""
    return f"{'FAILED':>{width}}" if v == INF else f"{v:>{width}.2f}"


def _improvement(base: float, other: float) -> float:
    """% improvement of `other` over `base`; nan if either run failed."""
    if base == INF or other == INF or base <= 0:
        return float('nan')
    return (base - other) / base * 100


# ── one test case ───────────────────────────────────────────────────────────

def _run_case(tc: dict, log: IssueLog, summary_rows: list[dict]):
    """Run every algorithm on one test case and print its report."""
    print(f"\n{'='*65}")
    print(f"  {tc['name']}")
    print(f"{'='*65}")

    dag     = tc['dag']
    network = tc['network']
    dyn     = tc['dynamic_network']

    cp     = _cp_min(dag)
    t_seq  = _sequential_time(dag)
    uptime = _mean_uptime(dyn, network)

    # ── planned schedules ──────────────────────────────────────────────
    print("\n  Planning schedules …")
    case = tc['name'].split(':')[0]
    sched_heft          = _plan(log, case, "HEFT",          calc_heft,  dag, network)
    sched_hepftStatic   = _plan(log, case, "HEPFT Static",  calc_hepft, dag, network, dyn, False)
    sched_hepftDynamic  = _plan(log, case, "HEPFT Dynamic", calc_hepft, dag, network, dyn, True)
    sched_reactive      = _plan(log, case, "Reactive",      simulate_reactive, dag, network, dyn)

    # ── simulate planned schedules on the real dynamic network ─────────
    def _replay(name, sched):
        reasons = {}
        sim = simulate_on_dynamic(dag, dyn, sched, reasons)
        _log_replay_failures(log, case, name, reasons)
        return sim

    sim_heft         = _replay("HEFT",          sched_heft)
    sim_hepftStatic  = _replay("HEPFT Static",  sched_hepftStatic)
    sim_hepftDynamic = _replay("HEPFT Dynamic", sched_hepftDynamic)
    sim_reactive     = sched_reactive   # reactive is already "actual"

    ms_plan_heft         = _makespan(sched_heft)
    ms_plan_hepftStatic  = _makespan(sched_hepftStatic)
    ms_plan_hepftDynamic = _makespan(sched_hepftDynamic)

    ms_sim_heft         = _makespan(sim_heft)
    ms_sim_hepftStatic  = _makespan(sim_hepftStatic)
    ms_sim_hepftDynamic = _makespan(sim_hepftDynamic)
    ms_sim_reactive     = _makespan(sim_reactive)

    # ── planned makespans ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Planned makespans")
    print(f"{'─'*65}")
    print(f"  {'Algorithm':<16} {'Planned':>10}")
    print(f"  {'-'*28}")
    print(f"  {'HEFT':<16} {_fmt(ms_plan_heft)}")
    print(f"  {'HEPFT Static':<16} {_fmt(ms_plan_hepftStatic)}")
    print(f"  {'HEPFT Dynamic':<16} {_fmt(ms_plan_hepftDynamic)}")
    print(f"  {'Reactive':<16} {'(n/a)':>10}")

    # ── simulated makespans ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Simulated makespans (real dynamic network)")
    print(f"{'─'*65}")
    print(f"  {'Algorithm':<16} {'Simulated':>10}  {'Robustness':>10}")
    print(f"  {'-'*38}")

    def _row(name, ms_sim, ms_plan=None):
        # robustness = simulated / planned (1.0 = ran exactly as planned)
        if ms_plan is None:
            rob_str = "n/a"
        elif ms_sim == INF:
            rob_str = "FAILED"
        else:
            rob_str = f"{ms_sim / ms_plan:.4f}" if ms_plan > 0 else "nan"
        print(f"  {name:<16} {_fmt(ms_sim)}  {rob_str:>10}")

    _row("HEFT",          ms_sim_heft,         ms_plan_heft)
    _row("HEPFT Static",  ms_sim_hepftStatic,  ms_plan_hepftStatic)
    _row("HEPFT Dynamic", ms_sim_hepftDynamic, ms_plan_hepftDynamic)
    _row("Reactive",      ms_sim_reactive)

    # ── winner ─────────────────────────────────────────────────────────
    candidates = {
        "HEFT":          ms_sim_heft,
        "HEPFT Static":  ms_sim_hepftStatic,
        "HEPFT Dynamic": ms_sim_hepftDynamic,
        "Reactive":      ms_sim_reactive,
    }
    best    = min(candidates.values())
    winners = [n for n, v in candidates.items() if abs(v - best) < 1e-6]
    others  = sorted((v, n) for n, v in candidates.items() if n not in winners)
    if best == INF:
        print("\n  Winner: none (every algorithm failed)")
    elif not others:
        print(f"\n  Winner: {' = '.join(winners)}  (all tied)")
    elif others[0][0] == INF:
        print(f"\n  Winner: {' = '.join(winners)}  (all others failed)")
    else:
        run_ms, runner = others[0]
        margin = run_ms - best
        print(f"\n  Winner: {' = '.join(winners)}  (beats runner-up {runner} by "
              f"{margin:.2f} / {margin / run_ms * 100:.1f}%)")

    # ── per-algorithm metrics ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Per-algorithm metrics")
    print(f"{'─'*65}")
    print(f"  CP_min                             : {cp:.2f}")
    print(f"  T_sequential                       : {t_seq:.2f}")
    print()
    print(f"  {'Algorithm':<14} {'SLR':>8}  {'Speedup':>9}")
    print(f"  {'-'*34}")
    for name, ms in candidates.items():
        if ms == INF:
            print(f"  {name:<14} {'FAILED':>8}  {'FAILED':>9}")
            continue
        slr     = ms / cp    if cp > 0 else float('nan')
        speedup = t_seq / ms if ms > 0 else float('nan')
        print(f"  {name:<14} {slr:>8.4f}  {speedup:>9.4f}")

    print(f"\n  Mean uptime per up-period (lower = more volatile) : {uptime:.2f}")
    for name, ms in [("HEPFT Static ", ms_sim_hepftStatic),
                     ("HEPFT Dynamic", ms_sim_hepftDynamic),
                     ("Reactive     ", ms_sim_reactive)]:
        print(f"  {name} improvement over HEFT : {_improvement(ms_sim_heft, ms):+.2f}%")

    summary_rows.append({
        'name':           tc['name'],
        'uptime':         uptime,
        'heft':           ms_sim_heft,
        'hepft_static':   ms_sim_hepftStatic,
        'hepft_dynamic':  ms_sim_hepftDynamic,
        'reactive':       ms_sim_reactive,
    })


# ── main ────────────────────────────────────────────────────────────────────

def main():
    from demo import get_test_cases
    test_cases = get_test_cases()

    summary_rows: list[dict] = []

    log = IssueLog()

    for tc in test_cases:
        try:
            _run_case(tc, log, summary_rows)
        except Exception as e:
            # Anything unexpected: log it and move on to the next test case
            case = tc['name'].split(':')[0]
            log.add(case, "-", "-", f"{type(e).__name__}: {e}")
            with open(log.path, 'a') as f:
                f.write(traceback.format_exc() + "\n")
            print(f"\n  ! {case} stopped early: {e}  (details in {log.path})")

    print(f"\n  {log.count} issue(s) logged to {log.path}")

    # ── cross-case summary ──────────────────────────────────────────────────
    if len(summary_rows) > 1:
        print(f"\n{'='*86}")
        print("  Cross-Case Summary  (simulated makespans, * = best per row)")
        print(f"{'='*86}")
        print(
            f"  {'Test Case':<38} {'Uptime':>7} "
            f"{'HEFT':>8} {'H-Static':>9} {'H-Dynamic':>10} {'React':>8}"
        )
        print(f"  {'-'*84}")
        for r in summary_rows:
            best = min(r['heft'], r['hepft_static'], r['hepft_dynamic'], r['reactive'])
            def fmt(v):
                if v == INF:
                    return f"{'FAIL':>8} "
                mark = '*' if abs(v - best) < 0.05 else ' '
                return f"{v:8.1f}{mark}"
            print(f"  {r['name']:<38} {r['uptime']:7.1f} "
                  f"{fmt(r['heft'])}{fmt(r['hepft_static'])} "
                  f"{fmt(r['hepft_dynamic'])}{fmt(r['reactive'])}")

        # Trend: improvement over HEFT vs volatility.
        # Uptime per up-period is LOWER for MORE volatile networks, so sort
        # from most stable (highest uptime) to most volatile (lowest).
        for label, key in [("HEPFT Static",  "hepft_static"),
                           ("HEPFT Dynamic", "hepft_dynamic"),
                           ("Reactive",      "reactive")]:
            print(f"\n{'─'*65}")
            print(f"  {label} improvement over HEFT  vs  Network Volatility")
            print(f"{'─'*65}")
            pairs = sorted(
                ((r['uptime'], _improvement(r['heft'], r[key])) for r in summary_rows),
                key=lambda x: -x[0],
            )
            for u, p in pairs:
                tag = ("a run failed" if p != p else
                       f"{label} better" if p >= 0 else "HEFT better")
                print(f"  uptime={u:7.1f}   {p:+7.2f}%   ({tag})")

            mid    = len(pairs) // 2
            stable = [p for _, p in pairs[:mid] if p == p]   # skip nan
            vol    = [p for _, p in pairs[mid:] if p == p]
            if not stable or not vol:
                print("\n  Not enough completed runs to compare volatility levels")
                continue
            low_avg  = sum(stable) / len(stable)
            high_avg = sum(vol)    / len(vol)
            print(f"\n  Avg improvement — low  volatility : {low_avg:+.2f}%")
            print(f"  Avg improvement — high volatility : {high_avg:+.2f}%")
            if high_avg > low_avg:
                print(f"  → {label}'s advantage grows with network volatility ✓")
            elif high_avg < low_avg:
                print(f"  → {label}'s advantage shrinks at higher volatility")
            else:
                print(f"  → No clear trend across volatility levels")


if __name__ == "__main__":
    main()