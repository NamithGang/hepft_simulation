class TaskDAG:
    def __init__(self):
        self.nodes = {}   # task_id: Task
        self.edges = {}   # (src, dst): communication_cost

    def compute_ranks(self, network, dynamic_network=None) -> dict[int, float]:
    
    # ─────────────────────────────────────────────
    # Pass 1: static rank using average bandwidth
    # (same as original — gives us estimated completion times)
    # ─────────────────────────────────────────────
    
        static_ranks = {}
        avg_bw = sum(network.bandwidth.values()) / len(network.bandwidth)

        for task_id in self._topological_sort():
            task = self.nodes[task_id]
            avg_comp = sum(task.comp_costs.values()) / len(task.comp_costs)

            if not task.children:
                static_ranks[task_id] = avg_comp
            else:
                max_successor = max(
                    (self.edges[(task_id, child)] / avg_bw) + static_ranks[child]
                    for child in task.children
                )
                static_ranks[task_id] = avg_comp + max_successor

        # if no dynamic network provided, return static ranks as-is
        if dynamic_network is None:
            return static_ranks

        # ─────────────────────────────────────────────
        # Pass 2: dynamic rank using predicted bandwidth
        # at each task's expected completion time
        # ─────────────────────────────────────────────

        dynamic_ranks = {}

        for task_id in self._topological_sort():
            task = self.nodes[task_id]
            avg_comp = sum(task.comp_costs.values()) / len(task.comp_costs)

            # use static rank as the estimated completion time for this task
            # this tells us when this task is likely to be done and
            # transferring data to its children
            estimated_completion = static_ranks[task_id]

            # get the predicted network state at that completion time
            predicted_net = dynamic_network.pred_net_func(estimated_completion)

            # compute average bandwidth of the predicted network at that time
            # fall back to static avg_bw if the predicted network has no links
            if predicted_net.bandwidth:
                predicted_avg_bw = sum(predicted_net.bandwidth.values()) / len(predicted_net.bandwidth)
            else:
                predicted_avg_bw = avg_bw

            if not task.children:
                dynamic_ranks[task_id] = avg_comp
            else:
                max_successor = max(
                    (self.edges[(task_id, child)] / predicted_avg_bw) + dynamic_ranks[child]
                    for child in task.children
                )
                dynamic_ranks[task_id] = avg_comp + max_successor

        return dynamic_ranks
    
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