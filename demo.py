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


def create_dag(n: int = 10, p: float = 0.3, seed: int = 42, num_processors: int = 3) -> TaskDAG:
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
# DYNAMIC: volatile WorkflowSim-style network
# ─────────────────────────────────────────────

class WorkflowSimNetwork:
    def __init__(self, base_network: NetworkGraph, seed: int = 42,
                 fluctuation_interval: tuple = (2, 8),
                 fluctuation_range: tuple = (0.2, 1.8),
                 failure_interval: tuple = (10, 25),
                 recovery_interval: tuple = (3, 10)):
        self.base_network = base_network
        self.rng = random.Random(seed)
        self.snapshots: list[tuple[float, NetworkGraph]] = [(0.0, base_network)]
        self.env = simpy.Environment()
        # configurable volatility parameters
        self.fluctuation_interval = fluctuation_interval   # how often bandwidth changes
        self.fluctuation_range = fluctuation_range         # how much bandwidth changes
        self.failure_interval = failure_interval           # how often failures occur
        self.recovery_interval = recovery_interval         # how long failures last

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
        while True:
            yield env.timeout(self.rng.uniform(*self.fluctuation_interval))
            for (src, dst) in list(current_bw.keys()):
                if src in active_procs and dst in active_procs:
                    base = self.base_network.bandwidth.get((src, dst), 1.0)
                    current_bw[(src, dst)] = round(
                        base * self.rng.uniform(*self.fluctuation_range), 3
                    )
            self.snapshots.append((env.now, self._make_snapshot(current_bw, active_procs)))

    def _processor_failure(self, env, current_bw: dict, active_procs: set):
        while True:
            yield env.timeout(self.rng.uniform(*self.failure_interval))
            if len(active_procs) <= 1:
                continue
            victim = self.rng.choice(list(active_procs))
            active_procs.discard(victim)
            self.snapshots.append((env.now, self._make_snapshot(current_bw, active_procs)))
            yield env.timeout(self.rng.uniform(*self.recovery_interval))
            active_procs.add(victim)
            self.snapshots.append((env.now, self._make_snapshot(current_bw, active_procs)))

    def run(self, until: float = 100.0) -> DynamicNetwork:
        current_bw = dict(self.base_network.bandwidth)
        active_procs = set(self.base_network.processors.keys())
        self.env.process(self._bandwidth_fluctuation(self.env, current_bw, active_procs))
        self.env.process(self._processor_failure(self.env, current_bw, active_procs))
        self.env.run(until=until)
        dyn = DynamicNetwork(self.base_network)
        for t, net in sorted(self.snapshots):
            if t > 0:
                dyn.add_snapshot(t, net)
        return dyn


# ─────────────────────────────────────────────
# Base network
# ─────────────────────────────────────────────

def create_network() -> NetworkGraph:
    net = NetworkGraph()
    net.processors[0] = Processor(0, speed=1.0)
    net.processors[1] = Processor(1, speed=1.5)
    net.processors[2] = Processor(2, speed=0.8)
    net.bandwidth = {
        (0, 1): 1.0, (1, 0): 1.0,
        (0, 2): 0.5, (2, 0): 0.5,
        (1, 2): 2.0, (2, 1): 2.0,
    }
    return net


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

def get_test_cases() -> list[dict]:
    """
    Returns 10 test cases, each with a dag, base network, and dynamic network.
    Volatility increases progressively across test cases.
    """
    base = create_network()

    return [
        # 1 — small stable network, sparse DAG
        {
            "name": "TC1: small sparse stable",
            "dag": create_dag(n=5, p=0.2, seed=1),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=1,
                fluctuation_interval=(10, 20),   # slow fluctuation
                fluctuation_range=(0.9, 1.1),    # tiny bandwidth change ±10%
                failure_interval=(80, 100),      # almost never fails
                recovery_interval=(1, 2)
            ).run(until=100.0)
        },

        # 2 — small stable network, dense DAG
        {
            "name": "TC2: small dense stable",
            "dag": create_dag(n=5, p=0.8, seed=2),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=2,
                fluctuation_interval=(10, 20),
                fluctuation_range=(0.9, 1.1),
                failure_interval=(80, 100),
                recovery_interval=(1, 2)
            ).run(until=100.0)
        },

        # 3 — medium DAG, moderate fluctuation, no failures
        {
            "name": "TC3: medium moderate fluctuation",
            "dag": create_dag(n=10, p=0.3, seed=3),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=3,
                fluctuation_interval=(5, 10),
                fluctuation_range=(0.6, 1.4),    # ±40% bandwidth change
                failure_interval=(90, 100),      # effectively no failures
                recovery_interval=(1, 2)
            ).run(until=100.0)
        },

        # 4 — medium DAG, moderate fluctuation with occasional failures
        {
            "name": "TC4: medium fluctuation + rare failures",
            "dag": create_dag(n=10, p=0.4, seed=4),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=4,
                fluctuation_interval=(5, 10),
                fluctuation_range=(0.6, 1.4),
                failure_interval=(30, 50),       # occasional failures
                recovery_interval=(5, 10)
            ).run(until=100.0)
        },

        # 5 — medium DAG, high fluctuation, no failures
        {
            "name": "TC5: medium high fluctuation",
            "dag": create_dag(n=10, p=0.5, seed=5),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=5,
                fluctuation_interval=(2, 5),     # fast fluctuation
                fluctuation_range=(0.3, 1.7),    # ±70% bandwidth change
                failure_interval=(90, 100),
                recovery_interval=(1, 2)
            ).run(until=100.0)
        },

        # 6 — medium DAG, high fluctuation + frequent failures
        {
            "name": "TC6: medium high fluctuation + frequent failures",
            "dag": create_dag(n=10, p=0.5, seed=6),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=6,
                fluctuation_interval=(2, 5),
                fluctuation_range=(0.3, 1.7),
                failure_interval=(15, 25),       # frequent failures
                recovery_interval=(3, 8)
            ).run(until=100.0)
        },

        # 7 — large DAG, moderate volatility
        {
            "name": "TC7: large moderate volatility",
            "dag": create_dag(n=20, p=0.3, seed=7),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=7,
                fluctuation_interval=(4, 8),
                fluctuation_range=(0.5, 1.5),
                failure_interval=(20, 40),
                recovery_interval=(4, 8)
            ).run(until=100.0)
        },

        # 8 — large dense DAG, high volatility
        {
            "name": "TC8: large dense high volatility",
            "dag": create_dag(n=20, p=0.6, seed=8),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=8,
                fluctuation_interval=(2, 4),
                fluctuation_range=(0.2, 1.8),    # ±80% bandwidth change
                failure_interval=(10, 20),       # very frequent failures
                recovery_interval=(2, 5)
            ).run(until=100.0)
        },

        # 9 — large DAG, extreme volatility, long recovery
        {
            "name": "TC9: large extreme volatility",
            "dag": create_dag(n=20, p=0.4, seed=9),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=9,
                fluctuation_interval=(1, 3),     # very fast fluctuation
                fluctuation_range=(0.1, 2.0),    # bandwidth can drop to 10% or double
                failure_interval=(8, 15),        # very frequent failures
                recovery_interval=(5, 15)        # long recovery time
            ).run(until=100.0)
        },

        # 10 — large dense DAG, worst case: near-constant churn
        {
            "name": "TC10: large dense worst case churn",
            "dag": create_dag(n=20, p=0.7, seed=10),
            "network": base,
            "dynamic_network": WorkflowSimNetwork(
                base, seed=10,
                fluctuation_interval=(1, 2),     # almost every time unit
                fluctuation_range=(0.05, 2.5),   # extreme bandwidth swings
                failure_interval=(5, 10),        # failures every 5-10 units
                recovery_interval=(8, 20)        # slow recovery
            ).run(until=100.0)
        },
    ]


# ─────────────────────────────────────────────
# Convenience factories (used by main)
# ─────────────────────────────────────────────

def create_dynamic_network(base_network: NetworkGraph) -> DynamicNetwork:
    sim = WorkflowSimNetwork(base_network, seed=42)
    return sim.run(until=100.0)