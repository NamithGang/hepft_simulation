from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from demo import create_dag, create_network, create_dynamic_network
from dynamic_network import DynamicNetwork
from heft import calc_heft
from hepft import calc_hepft
from cpop import calc_cpop
from reactive import simulate_reactive
from oracle_ilp import calc_oracle_ilp
 
 
# ── simulation helpers ──────────────────────────────────────────────────────
 
def simulate_on_dynamic(dag, dynamic_net, schedule):
    """
    Re-play a *planned* schedule against the true dynamic network and return
    the actual start/finish times.
 
    schedule: {task_id: (proc_id, planned_start, planned_finish)}
    returns : {task_id: (proc_id, actual_start, actual_finish)}
    """
    ordered = sorted(schedule.items(), key=lambda item: item[1][1])
 
    actual         = {}
    proc_available = {pid: 0.0 for pid in dynamic_net.pred_net_func(0.0).processors}
 
    for task_id, (proc_id, _, _) in ordered:
        ready_time = proc_available.get(proc_id, 0.0)
 
        for parent_id in dag.nodes[task_id].parents:
            parent_proc, _, parent_finish = actual[parent_id]
            net_at_t  = dynamic_net.pred_net_func(parent_finish)
            data_size = dag.edges[(parent_id, task_id)]
            comm = net_at_t.comm_cost(
                parent_proc, proc_id, data_size,
                fallback_bandwidth=dynamic_net.base_network.bandwidth
            )
            ready_time = max(ready_time, parent_finish + comm)
 
        start  = ready_time
        finish = start + dag.nodes[task_id].comp_costs[proc_id]
        actual[task_id]          = (proc_id, start, finish)
        proc_available[proc_id]  = finish
 
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
 
 
def _mean_volatility(dynamic_net: DynamicNetwork, network: NetworkGraph) -> float:
    vols = [dynamic_net.proc_volatility(pid) for pid in network.processors]
    return sum(vols) / len(vols) if vols else 0.0
 
 
# ── main ────────────────────────────────────────────────────────────────────
 
def main():
    from demo import get_test_cases
    test_cases = get_test_cases()
 
    summary_rows: list[dict] = []
 
    for tc in test_cases:
        print(f"\n{'='*65}")
        print(f"  {tc['name']}")
        print(f"{'='*65}")
 
        dag     = tc['dag']
        network = tc['network']
        dyn     = tc['dynamic_network']
 
        cp    = _cp_min(dag)
        t_seq = _sequential_time(dag)
        vol   = _mean_volatility(dyn, network)
 
        # ── planned schedules ──────────────────────────────────────────────
        print("\n  Planning schedules …")
        sched_heft     = calc_heft(dag, network)
        sched_cpop     = calc_cpop(dag, network)
        sched_hepft    = calc_hepft(dag, network, dyn)
        sched_reactive = simulate_reactive(dag, network, dyn)  # event-driven
 
        print("  Computing oracle lower bound …")
        oracle          = calc_oracle_ilp(dag, network, time_limit=60)
        oracle_makespan = oracle['makespan']
        oracle_method   = oracle['method']
        oracle_gap      = oracle['gap']
 
        # ── simulate planned schedules on the real dynamic network ─────────
        sim_heft     = simulate_on_dynamic(dag, dyn, sched_heft)
        sim_cpop     = simulate_on_dynamic(dag, dyn, sched_cpop)
        sim_hepft    = simulate_on_dynamic(dag, dyn, sched_hepft)
        sim_reactive = sched_reactive   # reactive is already "actual"
 
        ms_plan_heft     = _makespan(sched_heft)
        ms_plan_cpop     = _makespan(sched_cpop)
        ms_plan_hepft    = _makespan(sched_hepft)
 
        ms_sim_heft      = _makespan(sim_heft)
        ms_sim_cpop      = _makespan(sim_cpop)
        ms_sim_hepft     = _makespan(sim_hepft)
        ms_sim_reactive  = _makespan(sim_reactive)
 
        # ── planned makespans ──────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("  Planned makespans (static network)")
        print(f"{'─'*65}")
        print(f"  {'Algorithm':<14} {'Planned':>10}")
        print(f"  {'-'*26}")
        print(f"  {'HEFT':<14} {ms_plan_heft:>10.2f}")
        print(f"  {'CPOP':<14} {ms_plan_cpop:>10.2f}")
        print(f"  {'HEPFT':<14} {ms_plan_hepft:>10.2f}")
        print(f"  {'Reactive':<14} {'(n/a)':>10}")
 
        # ── simulated makespans ────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("  Simulated makespans (real dynamic network)")
        print(f"{'─'*65}")
        print(f"  {'Algorithm':<14} {'Simulated':>10}  {'vs Oracle':>10}  {'Robustness':>12}")
        print(f"  {'-'*50}")
 
        def _row(name, ms_sim, ms_plan=None):
            ratio_oracle = ms_sim / oracle_makespan if oracle_makespan > 0 else float('nan')
            robustness   = (ms_sim / ms_plan if ms_plan and ms_plan > 0
                            else float('nan'))
            rob_str = f"{robustness:.4f}" if ms_plan else "  n/a  "
            print(f"  {name:<14} {ms_sim:>10.2f}  {ratio_oracle:>10.4f}  {rob_str:>12}")
 
        _row("HEFT",     ms_sim_heft,     ms_plan_heft)
        _row("CPOP",     ms_sim_cpop,     ms_plan_cpop)
        _row("HEPFT",    ms_sim_hepft,    ms_plan_hepft)
        _row("Reactive", ms_sim_reactive)
 
        gap_str = f"  gap={oracle_gap*100:.1f}%" if oracle_gap > 0 else ""
        print(f"\n  Oracle lower bound  : {oracle_makespan:.2f}  [{oracle_method}]{gap_str}")
 
        # ── winner ─────────────────────────────────────────────────────────
        candidates = {
            "HEFT":     ms_sim_heft,
            "CPOP":     ms_sim_cpop,
            "HEPFT":    ms_sim_hepft,
            "Reactive": ms_sim_reactive,
        }
        ranking = sorted(candidates.items(), key=lambda kv: kv[1])
        winner, win_ms  = ranking[0]
        runner, run_ms  = ranking[1]
        margin = run_ms - win_ms
        print(f"\n  Winner: {winner}  (beats runner-up {runner} by "
              f"{margin:.2f} / {margin/run_ms*100:.1f}%)")
 
        # ── per-algorithm metrics ──────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("  Per-algorithm metrics")
        print(f"{'─'*65}")
        print(f"  CP_min                             : {cp:.2f}")
        print(f"  Oracle lower bound ({oracle_method:<16}): {oracle_makespan:.2f}")
        print(f"  T_sequential                       : {t_seq:.2f}")
        print()
        print(f"  {'Algorithm':<12} {'SLR':>8}  {'Speedup':>9}  {'Opt-gap%':>10}")
        print(f"  {'-'*44}")
        for name, ms in candidates.items():
            slr     = ms / cp    if cp    > 0 else float('nan')
            speedup = t_seq / ms if ms    > 0 else float('nan')
            opt_gap = (ms - oracle_makespan) / oracle_makespan * 100 if oracle_makespan > 0 else float('nan')
            print(f"  {name:<12} {slr:>8.4f}  {speedup:>9.4f}  {opt_gap:>+10.2f}%")
 
        print(f"\n  Network volatility (mean CV)       : {vol:.4f}")
        pct_hepft = (ms_sim_heft - ms_sim_hepft)    / ms_sim_heft * 100 if ms_sim_heft > 0 else 0.0
        pct_cpop  = (ms_sim_heft - ms_sim_cpop)     / ms_sim_heft * 100 if ms_sim_heft > 0 else 0.0
        pct_react = (ms_sim_heft - ms_sim_reactive)  / ms_sim_heft * 100 if ms_sim_heft > 0 else 0.0
        print(f"  CPOP     improvement over HEFT     : {pct_cpop:+.2f}%")
        print(f"  HEPFT    improvement over HEFT     : {pct_hepft:+.2f}%")
        print(f"  Reactive improvement over HEFT     : {pct_react:+.2f}%")
 
        summary_rows.append({
            'name':          tc['name'],
            'vol':           vol,
            'heft':          ms_sim_heft,
            'cpop':          ms_sim_cpop,
            'hepft':         ms_sim_hepft,
            'reactive':      ms_sim_reactive,
            'oracle':        oracle_makespan,
            'oracle_method': oracle_method,
        })
 
    # ── cross-case summary ──────────────────────────────────────────────────
    if len(summary_rows) > 1:
        print(f"\n{'='*75}")
        print("  Cross-Case Summary  (simulated makespans, * = best per row)")
        print(f"{'='*75}")
        print(f"  {'Test Case':<38} {'Vol':>5}  {'HEFT':>7}  {'CPOP':>7}  "
              f"{'HEPFT':>7}  {'React':>7}  {'Oracle':>8}")
        print(f"  {'-'*72}")
        for r in summary_rows:
            best = min(r['heft'], r['cpop'], r['hepft'], r['reactive'])
            def fmt(v):
                mark = '*' if abs(v - best) < 0.05 else ' '
                return f"{v:7.1f}{mark}"
            print(f"  {r['name']:<38} {r['vol']:5.3f}  "
                  f"{fmt(r['heft'])}  {fmt(r['cpop'])}  {fmt(r['hepft'])}  "
                  f"{fmt(r['reactive'])}  {r['oracle']:6.1f}({r['oracle_method'][:3]})")
 
        # Trend: HEPFT improvement vs volatility
        print(f"\n{'─'*65}")
        print("  HEPFT improvement over HEFT  vs  Network Volatility")
        print(f"{'─'*65}")
        pairs = sorted(
            (r['vol'], (r['heft'] - r['hepft']) / r['heft'] * 100)
            for r in summary_rows
        )
        mid      = len(pairs) // 2
        low_avg  = sum(p for _, p in pairs[:mid])  / max(mid, 1)
        high_avg = sum(p for _, p in pairs[mid:])  / max(len(pairs) - mid, 1)
        for v, p in pairs:
            tag = "HEPFT better" if p >= 0 else "HEFT  better"
            print(f"  vol={v:.3f}   {p:+.2f}%   ({tag})")
        print(f"\n  Avg improvement — low  volatility : {low_avg:+.2f}%")
        print(f"  Avg improvement — high volatility : {high_avg:+.2f}%")
        if high_avg > low_avg:
            print("  → HEPFT's advantage grows with network volatility ✓")
        elif high_avg < low_avg:
            print("  → HEPFT's advantage shrinks at higher volatility")
        else:
            print("  → No clear trend across volatility levels")
 
 
if __name__ == "__main__":
    main()