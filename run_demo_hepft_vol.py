from make_dag import TaskDAG
from make_network import NetworkGraph
from hepft_vol import calc_hepft_vol


# ── simulation helpers ──────────────────────────────────────────────────────

def simulate_on_dynamic(dag, dynamic_net, schedule):
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
        actual[task_id]         = (proc_id, start, finish)
        proc_available[proc_id] = finish

    return actual


def _makespan(schedule: dict) -> float:
    if not schedule:
        return float('inf')
    return (max(v[2] for v in schedule.values())
            - min(v[1] for v in schedule.values()))


# ── threshold combinations to test ─────────────────────────────────────────

PARAM_GRID = [
    (vt, at)
    for vt in [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0, 2.0]
    for at in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
]


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

        # ── run all parameter combinations ─────────────────────────────────
        print(f"\n  {'vol_thresh':>10}  {'avail_thresh':>12}  {'Simulated':>10}")
        print(f"  {'-'*36}")

        results = []
        for vol_thresh, avail_thresh in PARAM_GRID:
            sched  = calc_hepft_vol(dag, network, dyn,
                                    volatility_threshold=vol_thresh,
                                    availability_threshold=avail_thresh)
            ms_sim = _makespan(simulate_on_dynamic(dag, dyn, sched))
            results.append((vol_thresh, avail_thresh, ms_sim))
            print(f"  {vol_thresh:>10.2f}  {avail_thresh:>12.2f}  {ms_sim:>10.2f}")

        # ── winner ─────────────────────────────────────────────────────────
        results.sort(key=lambda r: r[2])
        best_vt, best_at, best_sim = results[0]
        print(f"\n  Best: vol_thresh={best_vt}, avail_thresh={best_at}  →  {best_sim:.2f}")

        summary_rows.append({
            'name':     tc['name'],
            'best_vt':  best_vt,
            'best_at':  best_at,
            'best_sim': best_sim,
        })

    # ── cross-case summary ──────────────────────────────────────────────────
    if len(summary_rows) > 1:
        print(f"\n{'='*65}")
        print("  Cross-Case Summary")
        print(f"{'='*65}")
        print(f"  {'Test Case':<30}  {'BestVT':>7}  {'BestAT':>7}  {'BestSim':>9}")
        print(f"  {'-'*52}")
        for r in summary_rows:
            print(f"  {r['name']:<30}  {r['best_vt']:>7.2f}  "
                  f"{r['best_at']:>7.2f}  {r['best_sim']:>9.2f}")


if __name__ == "__main__":
    main()