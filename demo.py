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
    """
    rng = random.Random(0)
    net = NetworkGraph()
    half = num_processors // 2

    for i in range(num_processors):
        # cluster A gets faster processors
        speed = rng.uniform(1.2, 1.8) if i < half else rng.uniform(0.6, 1.0)
        net.processors[i] = Processor(i, speed=round(speed, 2))

    for i in range(num_processors):
        for j in range(num_processors):
            if i == j:
                continue
            same_cluster = (i < half) == (j < half)
            # intra-cluster: high bandwidth; inter-cluster: lower bandwidth
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
        # cluster_size defines how many processors go down together
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
        Models events like a switch failure or rack power outage.
        """
        all_procs = list(self.base_network.processors.keys())
        while True:
            # wait longer between correlated failures — they're rarer but bigger
            yield env.timeout(self.rng.uniform(
                self.failure_interval[1],
                self.failure_interval[1] * 2
            ))

            # pick a random cluster of cluster_size processors to fail together
            candidates = [p for p in active_procs]
            if len(candidates) < 2:
                continue
            cluster_count = min(self.cluster_size, len(candidates) - 1)
            victims = self.rng.sample(candidates, cluster_count)

            # take the whole cluster down at once
            for v in victims:
                active_procs.discard(v)
            self.snapshots.append(
                (env.now, self._make_snapshot(current_bw, active_procs))
            )

            # recover them all at once after a longer outage
            yield env.timeout(self.rng.uniform(
                self.recovery_interval[0],
                self.recovery_interval[1]
            ))
            for v in victims:
                active_procs.add(v)
            self.snapshots.append(
                (env.now, self._make_snapshot(current_bw, active_procs))
            )

    def run(self, until: float = 200.0) -> DynamicNetwork:
        current_bw  = dict(self.base_network.bandwidth)
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
    10 test cases with increasing volatility, larger DAGs,
    more processors, and correlated failures.
    """

    return [
        # ── TC1: small, stable, 8 processors ──────────────────────────────
        {
            "name": "TC1: small sparse stable (8 procs)",
            "dag": create_dag(n=50, p=0.2, seed=1, num_processors=8),
            "network": create_network(num_processors=8),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=8), seed=1,
                fluctuation_interval=(15, 25),
                fluctuation_range=(0.9, 1.1),
                failure_interval=(80, 120),
                recovery_interval=(5, 10),
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.6,
            "availability_threshold": 0.6,
        },

        # ── TC2: small dense, stable, 8 processors ────────────────────────
        {
            "name": "TC2: small dense stable (8 procs)",
            "dag": create_dag(n=50, p=0.6, seed=2, num_processors=8),
            "network": create_network(num_processors=8),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=8), seed=2,
                fluctuation_interval=(15, 25),
                fluctuation_range=(0.9, 1.1),
                failure_interval=(80, 120),
                recovery_interval=(5, 10),
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.6,
            "availability_threshold": 0.6,
        },

        # ── TC3: medium DAG, moderate fluctuation, 8 processors ───────────
        {
            "name": "TC3: medium moderate fluctuation (8 procs)",
            "dag": create_dag(n=100, p=0.3, seed=3, num_processors=8),
            "network": create_network(num_processors=8),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=8), seed=3,
                fluctuation_interval=(5, 10),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(20, 40),
                recovery_interval=(10, 20),
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.5,
            "availability_threshold": 0.7,
        },

        # ── TC4: medium DAG, moderate fluctuation + failures, 8 procs ─────
        {
            "name": "TC4: medium fluctuation + failures (8 procs)",
            "dag": create_dag(n=100, p=0.4, seed=4, num_processors=8),
            "network": create_network(num_processors=8),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=8), seed=4,
                fluctuation_interval=(5, 10),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(10, 20),
                recovery_interval=(10, 25),
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.5,
            "availability_threshold": 0.7,
        },

        # ── TC5: medium DAG, high fluctuation, 12 processors ──────────────
        {
            "name": "TC5: medium high fluctuation (12 procs)",
            "dag": create_dag(n=150, p=0.3, seed=5, num_processors=12),
            "network": create_network(num_processors=12),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=12), seed=5,
                fluctuation_interval=(2, 6),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(10, 20),
                recovery_interval=(10, 30),
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.5,
            "availability_threshold": 0.7,
        },

        # ── TC6: medium DAG, correlated failures, 12 processors ───────────
        {
            "name": "TC6: medium correlated failures (12 procs)",
            "dag": create_dag(n=150, p=0.4, seed=6, num_processors=12),
            "network": create_network(num_processors=12),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=12), seed=6,
                fluctuation_interval=(3, 8),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(10, 20),
                recovery_interval=(10, 30),
                enable_correlated_failures=True,
                cluster_size=3
            ).run(until=200.0),
            "volatility_threshold": 0.45,
            "availability_threshold": 0.75,
        },

        # ── TC7: large DAG, moderate volatility, 12 processors ────────────
        {
            "name": "TC7: large moderate volatility (12 procs)",
            "dag": create_dag(n=250, p=0.2, seed=7, num_processors=12),
            "network": create_network(num_processors=12),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=12), seed=7,
                fluctuation_interval=(3, 7),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(10, 20),
                recovery_interval=(10, 30),
                enable_correlated_failures=True,
                cluster_size=4
            ).run(until=200.0),
            "volatility_threshold": 0.45,
            "availability_threshold": 0.75,
        },

        # ── TC8: large dense DAG, high volatility, 16 processors ──────────
        {
            "name": "TC8: large dense high volatility (16 procs)",
            "dag": create_dag(n=300, p=0.3, seed=8, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=8,
                fluctuation_interval=(2, 5),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(10, 20),
                recovery_interval=(15, 30),
                enable_correlated_failures=True,
                cluster_size=4
            ).run(until=200.0),
            "volatility_threshold": 0.4,
            "availability_threshold": 0.8,
        },

        # ── TC9: large DAG, extreme volatility, 16 processors ─────────────
        {
            "name": "TC9: large extreme volatility (16 procs)",
            "dag": create_dag(n=400, p=0.2, seed=9, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=9,
                fluctuation_interval=(1, 3),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(10, 20),
                recovery_interval=(15, 30),
                enable_correlated_failures=True,
                cluster_size=6
            ).run(until=200.0),
            "volatility_threshold": 0.4,
            "availability_threshold": 0.8,
        },

        # ── TC10: largest DAG, worst-case churn, 16 processors ────────────
        {
            "name": "TC10: largest dense worst case (16 procs)",
            "dag": create_dag(n=500, p=0.3, seed=10, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=10,
                fluctuation_interval=(1, 2),
                fluctuation_range=(0.2, 1.8),
                failure_interval=(10, 20),
                recovery_interval=(15, 30),
                enable_correlated_failures=True,
                cluster_size=8                   # half the cluster goes down at once
            ).run(until=200.0),
            "volatility_threshold": 0.35,
            "availability_threshold": 0.85,
        },

        # ── TC11: rapid bandwidth thrash, short recovery, 8 procs ──────────
        # Very fast fluctuations (every 0.5–1s) with extreme range make
        # bandwidth almost unpredictable; failures are brief but frequent.
        {
            "name": "TC11: rapid bandwidth thrash (8 procs)",
            "dag": create_dag(n=80, p=0.3, seed=11, num_processors=8),
            "network": create_network(num_processors=8),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=8), seed=11,
                fluctuation_interval=(0.5, 1.0),
                fluctuation_range=(0.05, 3.0),
                failure_interval=(5, 10),
                recovery_interval=(1, 3),
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.3,
            "availability_threshold": 0.85,
        },

        # ── TC12: near-permanent processor outages, 12 procs ───────────────
        # Failures happen very frequently and recovery takes a long time,
        # meaning processors spend more time down than up.
        {
            "name": "TC12: near-permanent outages (12 procs)",
            "dag": create_dag(n=120, p=0.25, seed=12, num_processors=12),
            "network": create_network(num_processors=12),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=12), seed=12,
                fluctuation_interval=(2, 4),
                fluctuation_range=(0.1, 2.0),
                failure_interval=(5, 8),
                recovery_interval=(40, 80),      # very long outages
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.3,
            "availability_threshold": 0.9,
        },

        # ── TC13: cascading correlated failures, 16 procs ──────────────────
        # Large cluster_size combined with rapid correlated failures means
        # the majority of processors can be simultaneously absent.
        {
            "name": "TC13: cascading correlated failures (16 procs)",
            "dag": create_dag(n=200, p=0.3, seed=13, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=13,
                fluctuation_interval=(1, 3),
                fluctuation_range=(0.1, 2.5),
                failure_interval=(8, 12),
                recovery_interval=(20, 40),
                enable_correlated_failures=True,
                cluster_size=10                  # 10 of 16 go down together
            ).run(until=200.0),
            "volatility_threshold": 0.25,
            "availability_threshold": 0.9,
        },

        # ── TC14: extreme bandwidth variance, sparse DAG, 12 procs ─────────
        # Bandwidth swings from near-zero to 5× base in very short windows,
        # making comm-cost prediction very unreliable.
        {
            "name": "TC14: extreme bandwidth variance sparse (12 procs)",
            "dag": create_dag(n=150, p=0.15, seed=14, num_processors=12),
            "network": create_network(num_processors=12),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=12), seed=14,
                fluctuation_interval=(0.5, 1.5),
                fluctuation_range=(0.02, 5.0),   # near-zero to 5× base
                failure_interval=(15, 25),
                recovery_interval=(5, 10),
                enable_correlated_failures=False
            ).run(until=200.0),
            "volatility_threshold": 0.25,
            "availability_threshold": 0.85,
        },

        # ── TC15: dense DAG + extreme churn, 16 procs ──────────────────────
        # High edge density means many comm-heavy dependencies; extreme
        # churn means none of those comm costs are stable.
        {
            "name": "TC15: dense DAG extreme churn (16 procs)",
            "dag": create_dag(n=250, p=0.5, seed=15, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=15,
                fluctuation_interval=(0.5, 1.0),
                fluctuation_range=(0.05, 4.0),
                failure_interval=(5, 10),
                recovery_interval=(10, 20),
                enable_correlated_failures=True,
                cluster_size=6
            ).run(until=200.0),
            "volatility_threshold": 0.25,
            "availability_threshold": 0.9,
        },

        # ── TC16: rolling wave failures, medium DAG, 12 procs ──────────────
        # Very short recovery windows mean processors cycle up and down
        # repeatedly, creating a "rolling wave" availability pattern.
        {
            "name": "TC16: rolling wave failures (12 procs)",
            "dag": create_dag(n=180, p=0.3, seed=16, num_processors=12),
            "network": create_network(num_processors=12),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=12), seed=16,
                fluctuation_interval=(1, 2),
                fluctuation_range=(0.1, 2.0),
                failure_interval=(3, 6),         # fail very often
                recovery_interval=(2, 4),        # recover quickly → many cycles
                enable_correlated_failures=True,
                cluster_size=4
            ).run(until=200.0),
            "volatility_threshold": 0.25,
            "availability_threshold": 0.9,
        },

        # ── TC17: asymmetric cluster churn, 16 procs ───────────────────────
        # One cluster (A) is rock-stable; cluster B churns aggressively.
        # Tests whether schedulers correctly migrate work to the stable side.
        {
            "name": "TC17: asymmetric cluster churn (16 procs)",
            "dag": create_dag(n=300, p=0.25, seed=17, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=17,
                fluctuation_interval=(0.5, 1.5),
                fluctuation_range=(0.05, 4.0),
                failure_interval=(4, 8),
                recovery_interval=(15, 30),
                enable_correlated_failures=True,
                cluster_size=8                   # entire second cluster goes down
            ).run(until=200.0),
            "volatility_threshold": 0.2,
            "availability_threshold": 0.9,
        },

        # ── TC18: micro-burst failures, large sparse DAG, 16 procs ─────────
        # Failures come in rapid micro-bursts (very short interval) but
        # each individual outage is also short — high churn, low down-time.
        {
            "name": "TC18: micro-burst failures sparse (16 procs)",
            "dag": create_dag(n=400, p=0.15, seed=18, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=18,
                fluctuation_interval=(0.3, 0.8),
                fluctuation_range=(0.1, 3.0),
                failure_interval=(2, 4),
                recovery_interval=(1, 2),        # short outages, very frequent
                enable_correlated_failures=True,
                cluster_size=5
            ).run(until=200.0),
            "volatility_threshold": 0.2,
            "availability_threshold": 0.9,
        },

        # ── TC19: maximum processors, catastrophic churn ───────────────────
        # Pushes to 16 procs with the most aggressive settings across all
        # dimensions: bandwidth, failures, cluster size, and DAG density.
        {
            "name": "TC19: maximum procs catastrophic churn (16 procs)",
            "dag": create_dag(n=450, p=0.35, seed=19, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=19,
                fluctuation_interval=(0.3, 0.7),
                fluctuation_range=(0.02, 5.0),
                failure_interval=(3, 5),
                recovery_interval=(20, 50),      # long recovery + frequent fail
                enable_correlated_failures=True,
                cluster_size=12                  # 12 of 16 go down together
            ).run(until=200.0),
            "volatility_threshold": 0.2,
            "availability_threshold": 0.95,
        },

        # ── TC20: worst-case everything, 16 procs ──────────────────────────
        # The absolute stress test: largest DAG, densest edges, fastest
        # fluctuations, most extreme bandwidth range, biggest cluster failures,
        # and longest recovery times.
        {
            "name": "TC20: worst-case everything (16 procs)",
            "dag": create_dag(n=500, p=0.4, seed=20, num_processors=16),
            "network": create_network(num_processors=16),
            "dynamic_network": WorkflowSimNetwork(
                create_network(num_processors=16), seed=20,
                fluctuation_interval=(0.2, 0.5),
                fluctuation_range=(0.01, 6.0),   # near-zero to 6× base
                failure_interval=(2, 4),
                recovery_interval=(30, 60),      # very long outages
                enable_correlated_failures=True,
                cluster_size=14                  # almost all procs go down
            ).run(until=200.0),
            "volatility_threshold": 0.15,
            "availability_threshold": 0.95,
        },
    ]


# ─────────────────────────────────────────────
# Convenience factories (used by simple main)
# ─────────────────────────────────────────────

def create_dynamic_network(base_network: NetworkGraph) -> DynamicNetwork:
    sim = WorkflowSimNetwork(base_network, seed=42)
    return sim.run(until=200.0)