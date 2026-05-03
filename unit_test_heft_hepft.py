import sys
import traceback
from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from dynamic_network import DynamicNetwork
from heft import calc_heft
from hepft import calc_hepft


# ── test harness ─────────────────────────────────────────────────────────────

_passed = 0
_failed = 0

def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  →  {detail}" if detail else ""))

def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ── graph / network builders ─────────────────────────────────────────────────

def make_single_task_graph(num_procs: int = 3):
    """One task, no edges. Each processor has a distinct comp cost."""
    dag = TaskDAG()
    dag.nodes[0] = Task(0, {p: float(p + 1) for p in range(num_procs)})
    return dag


def make_linear_chain(n: int = 3, comm: float = 1.0, num_procs: int = 2):
    """0 → 1 → … → n-1, all comp costs = 1, uniform comm cost."""
    dag = TaskDAG()
    for i in range(n):
        dag.nodes[i] = Task(i, {p: 1.0 for p in range(num_procs)})
    for i in range(n - 1):
        dag.add_edge(i, i + 1, comm_cost=comm)
    return dag


def make_fork_join(comm: float = 1.0):
    """
    Classic fork-join:
        0 → 1
        0 → 2
        1 → 3
        2 → 3
    comp costs: T0=1, T1=2, T2=3, T3=1
    """
    dag = TaskDAG()
    dag.nodes[0] = Task(0, {0: 1.0, 1: 1.0})
    dag.nodes[1] = Task(1, {0: 2.0, 1: 4.0})
    dag.nodes[2] = Task(2, {0: 4.0, 1: 3.0})
    dag.nodes[3] = Task(3, {0: 1.0, 1: 1.0})
    dag.add_edge(0, 1, comm_cost=comm)
    dag.add_edge(0, 2, comm_cost=comm)
    dag.add_edge(1, 3, comm_cost=comm)
    dag.add_edge(2, 3, comm_cost=comm)
    return dag


def make_network(num_procs: int = 2, bandwidth: float = 1.0) -> NetworkGraph:
    net = NetworkGraph()
    for p in range(num_procs):
        net.processors[p] = Processor(p)
    for i in range(num_procs):
        for j in range(num_procs):
            if i != j:
                net.bandwidth[(i, j)] = bandwidth
    return net


def make_static_dynamic(network: NetworkGraph) -> DynamicNetwork:
    """A DynamicNetwork that never changes — used to test HEPFT on stable nets."""
    return DynamicNetwork(network)


def make_dynamic_with_proc_down(network: NetworkGraph,
                                proc_id: int,
                                t_down: float,
                                t_up: float) -> DynamicNetwork:
    """
    Returns a DynamicNetwork where proc_id is absent between t_down and t_up.
    """
    dyn = DynamicNetwork(network)

    # snapshot at t_down: remove proc
    net_down = NetworkGraph()
    for pid, p in network.processors.items():
        if pid != proc_id:
            net_down.processors[pid] = p
    for (src, dst), bw in network.bandwidth.items():
        if src != proc_id and dst != proc_id:
            net_down.bandwidth[(src, dst)] = bw
    dyn.add_snapshot(t_down, net_down)

    # snapshot at t_up: restore proc
    dyn.add_snapshot(t_up, network)

    return dyn


# ── HEFT tests ───────────────────────────────────────────────────────────────

def test_heft_single_task():
    """HEFT on one task: must pick the fastest processor (P0, cost=1)."""
    print("\n[HEFT] Single task")
    dag = make_single_task_graph(num_procs=3)
    net = make_network(num_procs=3, bandwidth=1.0)
    sched = calc_heft(dag, net)

    check("task 0 scheduled", 0 in sched)
    proc, est, eft = sched[0]
    check("picks fastest processor (P0)", proc == 0,
          f"got P{proc}")
    check("EST = 0", close(est, 0.0), f"EST={est}")
    check("EFT = 1 (comp cost of P0)", close(eft, 1.0), f"EFT={eft}")


def test_heft_two_tasks_same_proc():
    """
    Two independent tasks where one processor is heavily preferred.
    Both should land on the best processor and execute sequentially.

    NOTE: compute_ranks() divides by len(network.bandwidth) which is 0 when
    there is only one processor (no links exist).  That is a real bug in
    make_dag.py — this test uses 2 processors to avoid triggering it, but
    makes P1 so slow that both tasks still prefer P0.
    """
    print("\n[HEFT] Two independent tasks forced onto same processor")
    dag = TaskDAG()
    dag.nodes[0] = Task(0, {0: 2.0, 1: 999.0})
    dag.nodes[1] = Task(1, {0: 3.0, 1: 999.0})
    net = make_network(num_procs=2, bandwidth=1.0)

    sched = calc_heft(dag, net)

    _, est0, eft0 = sched[0]
    _, est1, eft1 = sched[1]
    finishes = sorted([eft0, eft1])

    check("total makespan = 5", close(finishes[-1], 5.0),
          f"makespan={finishes[-1]}")
    check("tasks don't overlap", eft0 <= est1 or eft1 <= est0,
          f"T0=[{est0},{eft0}] T1=[{est1},{eft1}]")


def test_heft_comm_cost_zero_on_same_proc():
    """
    Chain T0→T1 with comm cost 10.  If both land on the same processor
    the comm cost is 0 — HEFT should prefer that over splitting them.
    """
    print("\n[HEFT] Chain — zero comm when same processor")
    dag = make_linear_chain(n=2, comm=10.0, num_procs=2)
    net = make_network(num_procs=2, bandwidth=1.0)   # bandwidth=1 ⟹ cost=10
    sched = calc_heft(dag, net)

    p0 = sched[0][0]
    p1 = sched[1][0]
    check("both tasks on same processor", p0 == p1,
          f"T0→P{p0}, T1→P{p1}")


def test_heft_respects_precedence():
    """T1 cannot start before T0 finishes (+ comm if different procs)."""
    print("\n[HEFT] Precedence constraint in chain")
    dag = make_linear_chain(n=3, comm=1.0, num_procs=2)
    net = make_network(num_procs=2, bandwidth=1.0)
    sched = calc_heft(dag, net)

    for child in [1, 2]:
        parent = child - 1
        p_par, _, eft_par = sched[parent]
        p_chi, est_chi, _ = sched[child]
        comm = 0.0 if p_par == p_chi else 1.0   # data_size=1, bw=1
        check(f"T{child} starts after T{parent} + comm",
              est_chi >= eft_par + comm - 1e-9,
              f"EST({child})={est_chi:.2f} EFT({parent})={eft_par:.2f} comm={comm}")


def test_heft_fork_join_makespan():
    """
    Fork-join with high comm cost: the critical path determines makespan.
    HEFT should produce a schedule no worse than sequential execution.
    """
    print("\n[HEFT] Fork-join makespan sanity")
    dag = make_fork_join(comm=0.5)
    net = make_network(num_procs=2, bandwidth=2.0)   # comm_cost = 0.5/2 = 0.25
    sched = calc_heft(dag, net)

    ms = max(v[2] for v in sched.values()) - min(v[1] for v in sched.values())
    seq = sum(min(t.comp_costs.values()) for t in dag.nodes.values())  # = 1+2+3+1=7
    check("makespan ≤ sequential time", ms <= seq + 1e-9,
          f"makespan={ms:.2f} sequential={seq:.2f}")
    check("all 4 tasks scheduled", len(sched) == 4, f"got {len(sched)}")


def test_heft_all_tasks_scheduled():
    """Every task in the DAG must appear in the schedule."""
    print("\n[HEFT] All tasks scheduled")
    from demo import create_dag, create_network
    dag = create_dag(n=20, p=0.3, seed=99, num_processors=4)
    net = create_network(num_processors=4)
    sched = calc_heft(dag, net)

    check("all tasks present", set(sched.keys()) == set(dag.nodes.keys()),
          f"missing={set(dag.nodes.keys()) - set(sched.keys())}")


def test_heft_no_overlap_on_same_processor():
    """Two tasks on the same processor must not overlap in time."""
    print("\n[HEFT] No processor overlap")
    from demo import create_dag, create_network
    dag = create_dag(n=30, p=0.4, seed=7, num_processors=3)
    net = create_network(num_processors=3)
    sched = calc_heft(dag, net)

    by_proc: dict[int, list] = {}
    for tid, (proc, est, eft) in sched.items():
        by_proc.setdefault(proc, []).append((est, eft, tid))

    overlap_found = False
    for proc, intervals in by_proc.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0] + 1e-9:
                overlap_found = True
                break

    check("no processor overlap", not overlap_found)


# ── HEPFT tests ──────────────────────────────────────────────────────────────

def test_hepft_stable_matches_heft():
    """
    On a completely static dynamic network (no changes), HEPFT should
    produce the same makespan as HEFT since there's nothing dynamic to react to.
    """
    print("\n[HEPFT] Static dynamic network → same makespan as HEFT")
    dag = make_fork_join(comm=0.5)
    net = make_network(num_procs=2, bandwidth=2.0)
    dyn = make_static_dynamic(net)

    sched_heft  = calc_heft(dag, net)
    sched_hepft = calc_hepft(dag, net, dyn)

    ms_heft  = max(v[2] for v in sched_heft.values())  - min(v[1] for v in sched_heft.values())
    ms_hepft = max(v[2] for v in sched_hepft.values()) - min(v[1] for v in sched_hepft.values())

    check("makespans match within 1e-6", abs(ms_heft - ms_hepft) < 1e-6,
          f"HEFT={ms_heft:.4f} HEPFT={ms_hepft:.4f}")


def test_hepft_avoids_down_processor():
    """
    P0 goes down at t=0.5 and recovers at t=100.  T0 has comp cost 5 on P0
    and 6 on P1.  HEPFT must NOT assign T0 to P0 because P0 will go down
    during execution (window [0, 5]).
    """
    print("\n[HEPFT] Avoids processor that is down during execution window")
    dag = TaskDAG()
    dag.nodes[0] = Task(0, {0: 5.0, 1: 6.0})

    net = make_network(num_procs=2, bandwidth=10.0)
    dyn = make_dynamic_with_proc_down(net, proc_id=0, t_down=0.5, t_up=100.0)

    sched = calc_hepft(dag, net, dyn)
    proc, est, eft = sched[0]

    check("T0 not assigned to P0 (it goes down during exec)",
          proc != 0, f"got P{proc}")
    check("T0 assigned to P1", proc == 1, f"got P{proc}")


def test_hepft_uses_down_proc_if_window_safe():
    """
    P0 goes down at t=50 (long after T0 would finish).
    HEPFT is free to use P0 since the execution window is safe.
    T0 comp cost: P0=2, P1=10 → P0 is much faster and safe.
    """
    print("\n[HEPFT] Uses processor that only fails after execution window")
    dag = TaskDAG()
    dag.nodes[0] = Task(0, {0: 2.0, 1: 10.0})

    net = make_network(num_procs=2, bandwidth=10.0)
    dyn = make_dynamic_with_proc_down(net, proc_id=0, t_down=50.0, t_up=100.0)

    sched = calc_hepft(dag, net, dyn)
    proc, _, eft = sched[0]

    check("T0 assigned to P0 (failure is far in the future)",
          proc == 0, f"got P{proc}")
    check("EFT = 2.0", close(eft, 2.0), f"EFT={eft}")


def test_hepft_all_tasks_scheduled():
    """Every task must appear in the HEPFT schedule."""
    print("\n[HEPFT] All tasks scheduled")
    from demo import create_dag, create_network
    dag = create_dag(n=20, p=0.3, seed=55, num_processors=4)
    net = create_network(num_processors=4)
    dyn = make_static_dynamic(net)
    sched = calc_hepft(dag, net, dyn)

    check("all tasks present", set(sched.keys()) == set(dag.nodes.keys()),
          f"missing={set(dag.nodes.keys()) - set(sched.keys())}")


def test_hepft_no_overlap_on_same_processor():
    """No two tasks on the same processor may overlap."""
    print("\n[HEPFT] No processor overlap")
    from demo import create_dag, create_network
    dag = create_dag(n=30, p=0.4, seed=13, num_processors=4)
    net = create_network(num_processors=4)
    dyn = make_static_dynamic(net)
    sched = calc_hepft(dag, net, dyn)

    by_proc: dict[int, list] = {}
    for tid, (proc, est, eft) in sched.items():
        by_proc.setdefault(proc, []).append((est, eft, tid))

    overlap_found = False
    for proc, intervals in by_proc.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0] + 1e-9:
                overlap_found = True
                break

    check("no processor overlap", not overlap_found)


def test_hepft_precedence_respected():
    """Child tasks must not start before their parents finish (+ comm)."""
    print("\n[HEPFT] Precedence constraints respected")
    dag = make_linear_chain(n=4, comm=1.0, num_procs=2)
    net = make_network(num_procs=2, bandwidth=1.0)
    dyn = make_static_dynamic(net)
    sched = calc_hepft(dag, net, dyn)

    for child in [1, 2, 3]:
        parent = child - 1
        p_par, _, eft_par = sched[parent]
        p_chi, est_chi, _ = sched[child]
        comm = 0.0 if p_par == p_chi else 1.0
        check(f"T{child} starts ≥ T{parent} finish + comm",
              est_chi >= eft_par + comm - 1e-9,
              f"EST({child})={est_chi:.3f} EFT({parent})+comm={eft_par+comm:.3f}")


def test_hepft_fallback_when_all_procs_down():
    """
    If both processors are down during a task's planned window, HEPFT
    must still schedule the task (fallback path) — not crash or skip it.
    """
    print("\n[HEPFT] Fallback when all processors are flagged as down")
    dag = TaskDAG()
    # Both procs are very slow so the window is long, both go down inside it
    dag.nodes[0] = Task(0, {0: 50.0, 1: 50.0})

    net = make_network(num_procs=2, bandwidth=10.0)

    # Take BOTH processors down at t=1 (before execution would finish)
    dyn = DynamicNetwork(net)
    net_empty = NetworkGraph()   # no processors
    dyn.add_snapshot(1.0, net_empty)
    dyn.add_snapshot(200.0, net)   # recover eventually

    sched = calc_hepft(dag, net, dyn)

    check("T0 is still scheduled (fallback used)", 0 in sched,
          "task missing from schedule")
    if 0 in sched:
        proc, est, eft = sched[0]
        check("scheduled on a valid processor", proc in (0, 1),
              f"got proc={proc}")


# ── run all tests ─────────────────────────────────────────────────────────────

def main():
    tests = [
        test_heft_single_task,
        test_heft_two_tasks_same_proc,
        test_heft_comm_cost_zero_on_same_proc,
        test_heft_respects_precedence,
        test_heft_fork_join_makespan,
        test_heft_all_tasks_scheduled,
        test_heft_no_overlap_on_same_processor,
        test_hepft_stable_matches_heft,
        test_hepft_avoids_down_processor,
        test_hepft_uses_down_proc_if_window_safe,
        test_hepft_all_tasks_scheduled,
        test_hepft_no_overlap_on_same_processor,
        test_hepft_precedence_respected,
        test_hepft_fallback_when_all_procs_down,
    ]

    for t in tests:
        try:
            t()
        except Exception:
            global _failed
            _failed += 1
            print(f"  ERROR in {t.__name__}:")
            traceback.print_exc()

    print(f"\n{'='*45}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print(f"{'='*45}")
    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
