# run_demo_toggle.py
"""
Tests HEPFT only, switching each of its three dynamic parts on and off:

  ranking        dynamic = rank on the time-averaged network
                 static  = rank on the starting network
  communication  dynamic = comm_cost_integrated (follows changing bandwidth)
                 static  = the starting network's comm_cost
  availability   dynamic = processors that join, must stay up for the whole
                           run, output must be able to leave
                 static  = every starting processor, always up

It runs eight configurations, from fully static to fully dynamic, on the same
test cases as run_demo.py, and scores each by replaying its schedule on the
real dynamic network (run_demo.simulate_on_dynamic). Problems are logged to
run_toggle_issues.txt and the run continues.
"""
from __future__ import annotations

import traceback

from demo     import get_test_cases
from hepft    import calc_hepft
from run_demo import (
    INF,
    IssueLog,
    _plan,
    _log_replay_failures,
    simulate_on_dynamic,
    _makespan,
    _fmt,
    _mean_uptime,
    _improvement,
)

ISSUE_LOG = "run_toggle_issues.txt"

# (label, dynamic_ranking, dynamic_comm, dynamic_avail), in test order:
# fully static, one dynamic part, two dynamic parts, fully dynamic
CONFIGS = [
    ("Static HEPFT",            False, False, False),
    ("Rank dyn",                True,  False, False),
    ("Comm dyn",                False, True,  False),
    ("Avail dyn",               False, False, True),
    ("Rank + Comm dyn",         True,  True,  False),
    ("Rank + Avail dyn",        True,  False, True),
    ("Comm + Avail dyn",        False, True,  True),
    ("Dynamic HEPFT",           True,  True,  True),
]


def _flags(rank: bool, comm: bool, avail: bool) -> str:
    """Short R/C/A code, e.g. 'D S S' = dynamic ranking, static comm and availability."""
    return " ".join("D" if f else "S" for f in (rank, comm, avail))


# ── one test case ───────────────────────────────────────────────────────────

def _run_case(tc: dict, log: IssueLog, summary_rows: list[dict]):
    """Run every HEPFT configuration on one test case and print its report."""
    print(f"\n{'='*72}")
    print(f"  {tc['name']}")
    print(f"{'='*72}")

    dag, network, dyn = tc['dag'], tc['network'], tc['dynamic_network']
    case = tc['name'].split(':')[0]

    print(f"\n  {'Configuration':<18} {'R C A':<6} {'Planned':>10} {'Simulated':>10} {'Robustness':>11}")
    print(f"  {'-'*59}")

    results = {}
    for label, rank, comm, avail in CONFIGS:
        sched = _plan(log, case, label, calc_hepft, dag, network,
                      dynamic_net=dyn, is_dynamic=True,
                      dynamic_ranking=rank, dynamic_comm=comm, dynamic_avail=avail)

        reasons = {}
        sim = simulate_on_dynamic(dag, dyn, sched, reasons)
        _log_replay_failures(log, case, label, reasons)

        ms_plan, ms_sim = _makespan(sched), _makespan(sim)
        if ms_sim == INF:
            rob = "FAILED"
        else:
            rob = f"{ms_sim / ms_plan:.4f}" if ms_plan > 0 else "nan"
        print(f"  {label:<18} {_flags(rank, comm, avail):<6} "
              f"{_fmt(ms_plan)} {_fmt(ms_sim)} {rob:>11}")
        results[label] = ms_sim

    # Best configuration and how each compares with fully static HEPFT
    best = min(results.values())
    if best == INF:
        print("\n  Best: none (every configuration failed)")
    else:
        winners = [n for n, v in results.items() if abs(v - best) < 1e-6]
        print(f"\n  Best: {' = '.join(winners)}  ({best:.2f})")

    static_ms = results["Static HEPFT"]
    print(f"\n  Improvement over static HEPFT (simulated makespan):")
    for label, *_ in CONFIGS[1:]:
        print(f"    {label:<18} {_improvement(static_ms, results[label]):+7.2f}%")

    summary_rows.append({'name': case, 'uptime': _mean_uptime(dyn, network), **results})


# ── main ────────────────────────────────────────────────────────────────────

def main():
    test_cases   = get_test_cases()
    summary_rows: list[dict] = []
    log = IssueLog(ISSUE_LOG)

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
    if not summary_rows:
        return

    # ── cross-case summary: simulated makespan per configuration ───────────
    labels = [c[0] for c in CONFIGS]
    codes  = [_flags(*c[1:]) for c in CONFIGS]
    print(f"\n{'='*(16 + 10 * len(labels))}")
    print("  Cross-Case Summary  (simulated makespans, * = best per row)")
    print("  Columns are R C A: ranking, communication, availability (D = dynamic, S = static)")
    print(f"{'='*(16 + 10 * len(labels))}")
    print(f"  {'Case':<6} {'Uptime':>7} " + "".join(f"{c:>10}" for c in codes))
    print(f"  {'-'*(14 + 10 * len(labels))}")
    for r in summary_rows:
        best = min(r[l] for l in labels)
        cells = ""
        for l in labels:
            v = r[l]
            if v == INF:
                cells += f"{'FAIL':>9} "
            else:
                cells += f"{v:9.1f}{'*' if abs(v - best) < 0.05 else ' '}"
        print(f"  {r['name']:<6} {r['uptime']:7.1f} {cells}")

    # ── average improvement over static HEPFT, and how often each is best ──
    print(f"\n{'─'*72}")
    print("  Average improvement over static HEPFT, and cases where each was best")
    print(f"{'─'*72}")
    print(f"  {'Configuration':<18} {'R C A':<6} {'Avg improvement':>16} {'Best in':>9}")
    for (label, *flags), code in zip(CONFIGS, codes):
        imps = [_improvement(r["Static HEPFT"], r[label]) for r in summary_rows]
        imps = [p for p in imps if p == p]                       # skip failed runs (nan)
        avg  = f"{sum(imps) / len(imps):+.2f}%" if imps else "n/a"
        wins = sum(1 for r in summary_rows
                   if r[label] != INF and abs(r[label] - min(r[l] for l in labels)) < 1e-6)
        print(f"  {label:<18} {code:<6} {avg:>16} {wins:>6}/{len(summary_rows)}")


if __name__ == "__main__":
    main()
