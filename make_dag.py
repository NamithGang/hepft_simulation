class TaskDAG:
    def __init__(self):
        self.nodes = {}   # task_id: Task
        self.edges = {}   # (src, dst): communication_cost

    def compute_ranks(self, network) -> dict[int, float]:
        ranks = {}
        avg_bw = sum(network.bandwidth.values()) / len(network.bandwidth)
        
        for task_id in self._topological_sort():
            task = self.nodes[task_id]
            avg_comp = sum(task.comp_costs.values()) / len(task.comp_costs)
            
            if not task.children:
                ranks[task_id] = avg_comp
            else:
                max_successor = max(
                    (self.edges[(task_id, child)] / avg_bw) + ranks[child]
                    for child in task.children
                )
                ranks[task_id] = avg_comp + max_successor
        
        return ranks
    def add_edge(self, src, dst, comm_cost):
        self.edges[(src, dst)] = comm_cost
        self.nodes[src].children.append(dst)
        self.nodes[dst].parents.append(src)

    def _topological_sort(self):
        visited = set()
        order = []

        def dfs(task_id):
            visited.add(task_id)
            for child in self.nodes[task_id].children:
                if child not in visited:
                    dfs(child)
            order.append(task_id)

        for task_id in self.nodes:
            if task_id not in visited:
                dfs(task_id)

        return order

class Task:
    def __init__(self, task_id, comp_costs):
        self.id = task_id
        self.comp_costs = comp_costs  # {proc_id: runtime}
        self.parents = []
        self.children = []