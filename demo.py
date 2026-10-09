# demo.py
import random
import simpy
import networkx as nx
from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from dynamic_network import DynamicNetwork


# ─────────────────────────────────────────────
# STATIC: NetworkX random DAG generator
# ─────────────────────────────────────────────

def generate_random_dag(n: int, p: float, seed: int = 42) -> nx.DiGraph:
    random.seed(seed)
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, weight=random.randint(1, 10))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j, weight=random.randint(1, 5))
    return G


def create_dag(n: int = 10, p: float = 0.3, seed: int = 42,
               num_processors: int = 3) -> TaskDAG:
    G = generate_random_dag(n, p, seed)
    dag = TaskDAG()
    rng = random.Random(seed)

    for node_id, data in G.nodes(data=True):
        base_cost = data['weight']
        comp_costs = {
            proc: round(base_cost * rng.uniform(0.5, 1.5), 2)
            for proc in range(num_processors)
        }
        dag.nodes[node_id] = Task(node_id, comp_costs)

    for src, dst, data in G.edges(data=True):
        dag.add_edge(src, dst, comm_cost=data['weight'])

    return dag


# ─────────────────────────────────────────────
# Network factory — supports 8-16 processors
# organised into clusters for correlated failures
# ─────────────────────────────────────────────

def create_network(num_processors: int = 8) -> NetworkGraph:
    """
    Creates a fully connected network with num_processors processors.
    Processors are split into two clusters:
      cluster A: procs 0 .. half-1       (faster, speed 1.2-1.8)
      cluster B: procs half .. n-1       (slower, speed 0.6-1.0)
    Intra-cluster bandwidth is higher than inter-cluster bandwidth.
    Uses a fixed seed of 0 so the same call always produces the same network.
    """
    rng = random.Random(0)
    net = NetworkGraph()
    half = num_processors // 2

    for i in range(num_processors):
        speed = rng.uniform(1.2, 1.8) if i < half else rng.uniform(0.6, 1.0)
        net.processors[i] = Processor(i, speed=round(speed, 2))

    for i in range(num_processors):
        for j in range(num_processors):
            if i == j:
                continue
            same_cluster = (i < half) == (j < half)
            if same_cluster:
                bw = rng.uniform(3.0, 6.0)
            else:
                bw = rng.uniform(0.5, 1.5)
            net.bandwidth[(i, j)] = round(bw, 2)

    return net


# ─────────────────────────────────────────────
# DYNAMIC: volatile WorkflowSim-style network
# ─────────────────────────────────────────────

class WorkflowSimNetwork:
    def __init__(self, base_network: NetworkGraph, seed: int = 42,
                 fluctuation_interval: tuple = (2, 8),
                 fluctuation_range: tuple = (0.2, 1.8),
                 failure_interval: tuple = (10, 20),
                 recovery_interval: tuple = (10, 30),
                 enable_correlated_failures: bool = False,
                 cluster_size: int = 4):
        self.base_network               = base_network
        self.rng                        = random.Random(seed)
        self.snapshots                  = [(0.0, base_network)]
        self.env                        = simpy.Environment()
        self.fluctuation_interval       = fluctuation_interval
        self.fluctuation_range          = fluctuation_range
        self.failure_interval           = failure_interval
        self.recovery_interval          = recovery_interval
        self.enable_correlated_failures = enable_correlated_failures
        self.cluster_size               = cluster_size

    def _make_snapshot(self, current_bw: dict, active_procs: set) -> NetworkGraph:
        net = NetworkGraph()
        for proc_id, proc in self.base_network.processors.items():
            if proc_id in active_procs:
                net.processors[proc_id] = Processor(proc_id, speed=proc.speed)
        for (src, dst), bw in current_bw.items():
            if src in active_procs and dst in active_procs:
                net.bandwidth[(src, dst)] = bw
        return net

    def _bandwidth_fluctuation(self, env, current_bw: dict, active_procs: set):
        """Randomly fluctuate bandwidth on all active links."""
        while True:
            yield env.timeout(self.rng.uniform(*self.fluctuation_interval))
            for (src, dst) in list(current_bw.keys()):
                if src in active_procs and dst in active_procs:
                    base = self.base_network.bandwidth.get((src, dst), 1.0)
                    current_bw[(src, dst)] = round(
                        base * self.rng.uniform(*self.fluctuation_range), 3
                    )
            self.snapshots.append(
                (env.now, self._make_snapshot(current_bw, active_procs))
            )

    def _single_processor_failure(self, env, current_bw: dict, active_procs: set):
        """Randomly take one processor offline then recover it."""
        while True:
            yield env.timeout(self.rng.uniform(*self.failure_interval))
            candidates = [p for p in active_procs]
            if not candidates:
                continue
            victim = self.rng.choice(candidates)
            active_procs.discard(victim)
            self.snapshots.append(
                (env.now, self._make_snapshot(current_bw, active_procs))
            )
            yield env.timeout(self.rng.uniform(*self.recovery_interval))
            active_procs.add(victim)
            self.snapshots.append(
                (env.now, self._make_snapshot(current_bw, active_procs))
            )

    def _correlated_failure(self, env, current_bw: dict, active_procs: set):
        """
        Simulate a network partition: take down a whole cluster of processors
        simultaneously, then recover them together.
        """
        while True:
            yield env.timeout(self.rng.uniform(
                self.failure_interval[1],
                self.failure_interval[1] * 2
            ))
            candidates = [p for p in active_procs]
            if len(candidates) < 2:
                continue
            cluster_count = min(self.cluster_size, len(candidates) - 1)
            victims = self.rng.sample(candidates, cluster_count)
            for v in victims:
                active_procs.discard(v)
            self.snapshots.append(
                (env.now, self._make_snapshot(current_bw, active_procs))
            )
            yield env.timeout(self.rng.uniform(
                self.recovery_interval[0],
                self.recovery_interval[1]
            ))
            for v in victims:
                active_procs.add(v)
            self.snapshots.append(
                (env.now, self._make_snapshot(current_bw, active_procs))
            )

    def run(self, until: float = 700.0) -> DynamicNetwork:
        current_bw   = dict(self.base_network.bandwidth)
        active_procs = set(self.base_network.processors.keys())

        self.env.process(
            self._bandwidth_fluctuation(self.env, current_bw, active_procs)
        )
        self.env.process(
            self._single_processor_failure(self.env, current_bw, active_procs)
        )
        if self.enable_correlated_failures:
            self.env.process(
                self._correlated_failure(self.env, current_bw, active_procs)
            )

        self.env.run(until=until)

        dyn = DynamicNetwork(self.base_network)
        for t, net in sorted(self.snapshots):
            if t > 0:
                dyn.add_snapshot(t, net)
        return dyn


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

def get_test_cases() -> list[dict]:
    """
    20 test cases with increasing volatility, larger DAGs,
    more processors, and correlated failures.

    Each test case reuses the same base_network object for both
    "network" and WorkflowSimNetwork — guaranteeing they start
    from identical topologies.
    """

    cases = []

    # ── TC1: small, stable, 8 processors ──────────────────────────────────
    _net = create_network(num_processors=8)
    cases.append({
        "name": "TC1: small sparse stable (8 procs)",
        "dag": create_dag(n=50, p=0.2, seed=1, num_processors=8),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=1,
            fluctuation_interval=(15, 25),
            fluctuation_range=(0.9, 1.1),
            failure_interval=(80, 120),
            recovery_interval=(5, 10),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.6,
        "availability_threshold": 0.6,
    })

    # ── TC2: small dense, stable, 8 processors ────────────────────────────
    _net = create_network(num_processors=8)
    cases.append({
        "name": "TC2: small dense stable (8 procs)",
        "dag": create_dag(n=50, p=0.6, seed=2, num_processors=8),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=2,
            fluctuation_interval=(15, 25),
            fluctuation_range=(0.9, 1.1),
            failure_interval=(80, 120),
            recovery_interval=(5, 10),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.6,
        "availability_threshold": 0.6,
    })

    # ── TC3: medium DAG, moderate fluctuation, 8 processors ───────────────
    _net = create_network(num_processors=8)
    cases.append({
        "name": "TC3: medium moderate fluctuation (8 procs)",
        "dag": create_dag(n=100, p=0.3, seed=3, num_processors=8),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=3,
            fluctuation_interval=(5, 10),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(20, 40),
            recovery_interval=(10, 20),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.5,
        "availability_threshold": 0.7,
    })

    # ── TC4: medium DAG, moderate fluctuation + failures, 8 procs ─────────
    _net = create_network(num_processors=8)
    cases.append({
        "name": "TC4: medium fluctuation + failures (8 procs)",
        "dag": create_dag(n=100, p=0.4, seed=4, num_processors=8),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=4,
            fluctuation_interval=(5, 10),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(10, 20),
            recovery_interval=(10, 25),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.5,
        "availability_threshold": 0.7,
    })

    # ── TC5: medium DAG, high fluctuation, 12 processors ──────────────────
    _net = create_network(num_processors=12)
    cases.append({
        "name": "TC5: medium high fluctuation (12 procs)",
        "dag": create_dag(n=150, p=0.3, seed=5, num_processors=12),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=5,
            fluctuation_interval=(2, 6),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(10, 20),
            recovery_interval=(10, 30),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.5,
        "availability_threshold": 0.7,
    })

    # ── TC6: medium DAG, correlated failures, 12 processors ───────────────
    _net = create_network(num_processors=12)
    cases.append({
        "name": "TC6: medium correlated failures (12 procs)",
        "dag": create_dag(n=150, p=0.4, seed=6, num_processors=12),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=6,
            fluctuation_interval=(3, 8),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(10, 20),
            recovery_interval=(10, 30),
            enable_correlated_failures=True,
            cluster_size=3
        ).run(until=700),
        "volatility_threshold": 0.45,
        "availability_threshold": 0.75,
    })

    # ── TC7: large DAG, moderate volatility, 12 processors ────────────────
    _net = create_network(num_processors=12)
    cases.append({
        "name": "TC7: large moderate volatility (12 procs)",
        "dag": create_dag(n=250, p=0.2, seed=7, num_processors=12),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=7,
            fluctuation_interval=(3, 7),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(10, 20),
            recovery_interval=(10, 30),
            enable_correlated_failures=True,
            cluster_size=4
        ).run(until=700),
        "volatility_threshold": 0.45,
        "availability_threshold": 0.75,
    })

    # ── TC8: large dense DAG, high volatility, 16 processors ──────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC8: large dense high volatility (16 procs)",
        "dag": create_dag(n=300, p=0.3, seed=8, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=8,
            fluctuation_interval=(2, 5),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(10, 20),
            recovery_interval=(15, 30),
            enable_correlated_failures=True,
            cluster_size=4
        ).run(until=700),
        "volatility_threshold": 0.4,
        "availability_threshold": 0.8,
    })

    # ── TC9: large DAG, extreme volatility, 16 processors ─────────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC9: large extreme volatility (16 procs)",
        "dag": create_dag(n=400, p=0.2, seed=9, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=9,
            fluctuation_interval=(1, 3),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(10, 20),
            recovery_interval=(15, 30),
            enable_correlated_failures=True,
            cluster_size=6
        ).run(until=700),
        "volatility_threshold": 0.4,
        "availability_threshold": 0.8,
    })

    # ── TC10: largest DAG, worst-case churn, 16 processors ────────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC10: largest dense worst case (16 procs)",
        "dag": create_dag(n=500, p=0.3, seed=10, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=10,
            fluctuation_interval=(1, 2),
            fluctuation_range=(0.2, 1.8),
            failure_interval=(10, 20),
            recovery_interval=(15, 30),
            enable_correlated_failures=True,
            cluster_size=8
        ).run(until=700),
        "volatility_threshold": 0.35,
        "availability_threshold": 0.85,
    })

    # ── TC11: rapid bandwidth thrash, short recovery, 8 procs ─────────────
    _net = create_network(num_processors=8)
    cases.append({
        "name": "TC11: rapid bandwidth thrash (8 procs)",
        "dag": create_dag(n=80, p=0.3, seed=11, num_processors=8),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=11,
            fluctuation_interval=(0.5, 1.0),
            fluctuation_range=(0.05, 3.0),
            failure_interval=(5, 10),
            recovery_interval=(1, 3),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.3,
        "availability_threshold": 0.85,
    })

    # ── TC12: near-permanent processor outages, 12 procs ──────────────────
    _net = create_network(num_processors=12)
    cases.append({
        "name": "TC12: near-permanent outages (12 procs)",
        "dag": create_dag(n=120, p=0.25, seed=12, num_processors=12),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=12,
            fluctuation_interval=(2, 4),
            fluctuation_range=(0.1, 2.0),
            failure_interval=(5, 8),
            recovery_interval=(40, 80),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.3,
        "availability_threshold": 0.9,
    })

    # ── TC13: cascading correlated failures, 16 procs ─────────────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC13: cascading correlated failures (16 procs)",
        "dag": create_dag(n=200, p=0.3, seed=13, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=13,
            fluctuation_interval=(1, 3),
            fluctuation_range=(0.1, 2.5),
            failure_interval=(8, 12),
            recovery_interval=(20, 40),
            enable_correlated_failures=True,
            cluster_size=10
        ).run(until=700),
        "volatility_threshold": 0.25,
        "availability_threshold": 0.9,
    })

    # ── TC14: extreme bandwidth variance, sparse DAG, 12 procs ────────────
    _net = create_network(num_processors=12)
    cases.append({
        "name": "TC14: extreme bandwidth variance sparse (12 procs)",
        "dag": create_dag(n=150, p=0.15, seed=14, num_processors=12),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=14,
            fluctuation_interval=(0.5, 1.5),
            fluctuation_range=(0.02, 5.0),
            failure_interval=(15, 25),
            recovery_interval=(5, 10),
            enable_correlated_failures=False
        ).run(until=700),
        "volatility_threshold": 0.25,
        "availability_threshold": 0.85,
    })

    # ── TC15: dense DAG + extreme churn, 16 procs ─────────────────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC15: dense DAG extreme churn (16 procs)",
        "dag": create_dag(n=250, p=0.5, seed=15, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=15,
            fluctuation_interval=(0.5, 1.0),
            fluctuation_range=(0.05, 4.0),
            failure_interval=(5, 10),
            recovery_interval=(10, 20),
            enable_correlated_failures=True,
            cluster_size=6
        ).run(until=700),
        "volatility_threshold": 0.25,
        "availability_threshold": 0.9,
    })

    # ── TC16: rolling wave failures, medium DAG, 12 procs ─────────────────
    _net = create_network(num_processors=12)
    cases.append({
        "name": "TC16: rolling wave failures (12 procs)",
        "dag": create_dag(n=180, p=0.3, seed=16, num_processors=12),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=16,
            fluctuation_interval=(1, 2),
            fluctuation_range=(0.1, 2.0),
            failure_interval=(3, 6),
            recovery_interval=(2, 4),
            enable_correlated_failures=True,
            cluster_size=4
        ).run(until=700),
        "volatility_threshold": 0.25,
        "availability_threshold": 0.9,
    })

    # ── TC17: asymmetric cluster churn, 16 procs ───────────────────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC17: asymmetric cluster churn (16 procs)",
        "dag": create_dag(n=300, p=0.25, seed=17, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=17,
            fluctuation_interval=(0.5, 1.5),
            fluctuation_range=(0.05, 4.0),
            failure_interval=(4, 8),
            recovery_interval=(15, 30),
            enable_correlated_failures=True,
            cluster_size=8
        ).run(until=700),
        "volatility_threshold": 0.2,
        "availability_threshold": 0.9,
    })

    # ── TC18: micro-burst failures, large sparse DAG, 16 procs ────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC18: micro-burst failures sparse (16 procs)",
        "dag": create_dag(n=400, p=0.15, seed=18, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=18,
            fluctuation_interval=(0.3, 0.8),
            fluctuation_range=(0.1, 3.0),
            failure_interval=(2, 4),
            recovery_interval=(1, 2),
            enable_correlated_failures=True,
            cluster_size=5
        ).run(until=700),
        "volatility_threshold": 0.2,
        "availability_threshold": 0.9,
    })

    # ── TC19: maximum processors, catastrophic churn ───────────────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC19: maximum procs catastrophic churn (16 procs)",
        "dag": create_dag(n=450, p=0.35, seed=19, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=19,
            fluctuation_interval=(0.3, 0.7),
            fluctuation_range=(0.02, 5.0),
            failure_interval=(3, 5),
            recovery_interval=(20, 50),
            enable_correlated_failures=True,
            cluster_size=12
        ).run(until=700),
        "volatility_threshold": 0.2,
        "availability_threshold": 0.95,
    })

    # ── TC20: worst-case everything, 16 procs ─────────────────────────────
    _net = create_network(num_processors=16)
    cases.append({
        "name": "TC20: worst-case everything (16 procs)",
        "dag": create_dag(n=500, p=0.4, seed=20, num_processors=16),
        "network": _net,
        "dynamic_network": WorkflowSimNetwork(
            _net, seed=20,
            fluctuation_interval=(0.2, 0.5),
            fluctuation_range=(0.01, 6.0),
            failure_interval=(2, 4),
            recovery_interval=(30, 60),
            enable_correlated_failures=True,
            cluster_size=14
        ).run(until=700),
        "volatility_threshold": 0.15,
        "availability_threshold": 0.95,
    })

    return cases


# ─────────────────────────────────────────────
# Convenience factories (used by simple main)
# ─────────────────────────────────────────────

def create_dynamic_network(base_network: NetworkGraph) -> DynamicNetwork:
    sim = WorkflowSimNetwork(base_network, seed=42)
    return sim.run(until=700)


# ─────────────────────────────────────────────
# Small fixed test case — 8 tasks, 8 processors
# ─────────────────────────────────────────────
#
# DAG structure (task IDs 0–7):
#
#              T0
#           /  |  \
#         T1  T2  T3
#         |\ /|   |
#         | X |   |
#         |/ \|   |
#         T4  T5  T6
#          \  |  /
#             T7
#
# Edges with communication costs:
#   T0→T1(2)  T0→T2(3)  T0→T3(1)
#   T1→T4(2)  T1→T5(3)
#   T2→T4(1)  T2→T5(2)
#   T3→T6(2)
#   T4→T7(3)  T5→T7(2)  T6→T7(1)

def create_test_case() -> dict:
    """
    Returns a dict with keys:
      dag, network, dynamic_network, name,
      volatility_threshold, availability_threshold

    Task IDs are zero-indexed (0–7) consistent with processor IDs (0–7).
    The same base_network object is passed to both "network" and
    WorkflowSimNetwork so they are guaranteed identical topologies.
    """
    NUM_PROCS = 8

    # ── DAG ──────────────────────────────────────────────────────────────
    dag = TaskDAG()

    # comp_costs[task_id][proc_id] — deliberately varied so HEFT makes
    # non-trivial choices (not all tasks prefer the same processor)
    comp_data = {
        0: [6, 5, 7, 8, 6, 5, 7, 6],
        1: [8, 7, 6, 7, 9, 8, 7, 8],
        2: [5, 6, 5, 6, 5, 7, 6, 5],
        3: [7, 8, 9, 7, 6, 7, 8, 7],
        4: [9, 8, 7, 8, 9, 8, 9, 8],
        5: [6, 5, 6, 7, 6, 5, 6, 7],
        6: [8, 7, 8, 6, 7, 8, 7, 8],
        7: [4, 5, 4, 5, 4, 5, 4, 5],
    }
    for tid, costs in comp_data.items():
        dag.nodes[tid] = Task(tid, {p: float(c) for p, c in enumerate(costs)})

    edges = [
        (0, 1, 2), (0, 2, 3), (0, 3, 1),
        (1, 4, 2), (1, 5, 3),
        (2, 4, 1), (2, 5, 2),
        (3, 6, 2),
        (4, 7, 3), (5, 7, 2), (6, 7, 1),
    ]
    for src, dst, comm in edges:
        dag.add_edge(src, dst, comm_cost=comm)

    # ── Network — shared base object ──────────────────────────────────────
    base_net = create_network(num_processors=NUM_PROCS)

    # ── Dynamic network — mild fluctuation, occasional single failure ──────
    dynamic_net = WorkflowSimNetwork(
        base_net,
        seed=7,
        fluctuation_interval=(10, 20),
        fluctuation_range=(0.8, 1.2),
        failure_interval=(30, 60),
        recovery_interval=(5, 15),
        enable_correlated_failures=False,
    ).run(until=700)

    return {
        "name": "Small fixed 8-task DAG on 8 processors",
        "dag": dag,
        "network": base_net,
        "dynamic_network": dynamic_net,
        "volatility_threshold": 0.6,
        "availability_threshold": 0.6,
    }