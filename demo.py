# demo.py
import random
import simpy
import networkx as nx
from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from dynamic_network import DynamicNetwork


# ─────────────────────────────────────────────
# STATIC: NetworkX layer-by-layer random DAG
# replicates daggen's generation model
# ─────────────────────────────────────────────

def _generate_nx_dag(
    n: int = 10,
    fat: float = 0.5,
    density: float = 0.5,
    ccr: float = 0.5,
    seed: int = 42
) -> nx.DiGraph:
    """
    Generate a random DAG using NetworkX with daggen-style parameters:
      n       - number of computation tasks
      fat     - controls max tasks per level (higher = wider/more parallel)
      density - probability of an edge between tasks in adjacent levels
      ccr     - communication-to-computation ratio (scales edge weights)
    """
    rng = random.Random(seed)
    G = nx.DiGraph()

    # determine tasks per level based on fat parameter
    max_per_level = max(1, int(fat * n))
    levels = []
    remaining = n

    while remaining > 0:
        level_size = rng.randint(1, min(max_per_level, remaining))
        levels.append(level_size)
        remaining -= level_size

    # assign task ids and computation costs level by level
    task_id = 0
    level_nodes = []
    for level in levels:
        nodes = []
        for _ in range(level):
            cost = rng.uniform(10, 50)
            G.add_node(task_id, computation=cost)
            nodes.append(task_id)
            task_id += 1
        level_nodes.append(nodes)

    # add edges between adjacent levels based on density
    for i in range(len(level_nodes) - 1):
        current_level = level_nodes[i]
        next_level = level_nodes[i + 1]

        for src in current_level:
            for dst in next_level:
                if rng.random() < density:
                    comp_cost = G.nodes[src]['computation']
                    comm_weight = comp_cost * ccr * rng.uniform(0.5, 1.5)
                    G.add_edge(src, dst, data=comm_weight)

        # ensure every node in next_level has at least one parent
        for dst in next_level:
            if G.in_degree(dst) == 0:
                src = rng.choice(current_level)
                comp_cost = G.nodes[src]['computation']
                comm_weight = comp_cost * ccr * rng.uniform(0.5, 1.5)
                G.add_edge(src, dst, data=comm_weight)

    return G


def create_dag(n: int = 10, seed: int = 42, num_processors: int = 3) -> TaskDAG:
    """Convert a NetworkX DAG into our TaskDAG format."""
    G = _generate_nx_dag(n=n, seed=seed)
    dag = TaskDAG()
    rng = random.Random(seed)

    for node_id, data in G.nodes(data=True):
        base_cost = data['computation']
        comp_costs = {
            p: round(base_cost * rng.uniform(0.5, 1.5), 2)
            for p in range(num_processors)
        }
        dag.nodes[node_id] = Task(node_id, comp_costs)

    for src, dst, data in G.edges(data=True):
        dag.add_edge(src, dst, comm_cost=round(data['data'], 2))

    return dag


# ─────────────────────────────────────────────
# DYNAMIC: WorkflowSim-style network via SimPy
# ─────────────────────────────────────────────

class WorkflowSimNetwork:
    def __init__(self, base_network: NetworkGraph, seed: int = 42):
        self.base_network = base_network
        self.rng = random.Random(seed)
        self.snapshots: list[tuple[float, NetworkGraph]] = [(0.0, base_network)]
        self.env = simpy.Environment()

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
        """Randomly fluctuate bandwidth to simulate contention."""
        while True:
            yield env.timeout(self.rng.uniform(5, 15))
            for (src, dst) in list(current_bw.keys()):
                if src in active_procs and dst in active_procs:
                    base = self.base_network.bandwidth.get((src, dst), 1.0)
                    current_bw[(src, dst)] = round(base * self.rng.uniform(0.6, 1.4), 3)
            self.snapshots.append((env.now, self._make_snapshot(current_bw, active_procs)))

    def _processor_failure(self, env, current_bw: dict, active_procs: set):
        """Randomly take a processor offline then recover it."""
        while True:
            yield env.timeout(self.rng.uniform(20, 50))
            if len(active_procs) <= 1:
                continue
            victim = self.rng.choice(list(active_procs))
            active_procs.discard(victim)
            self.snapshots.append((env.now, self._make_snapshot(current_bw, active_procs)))
            yield env.timeout(self.rng.uniform(5, 15))
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
# Convenience factories
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

def create_dynamic_network(base_network: NetworkGraph) -> DynamicNetwork:
    sim = WorkflowSimNetwork(base_network, seed=42)
    return sim.run(until=100.0)