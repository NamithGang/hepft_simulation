from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from dynamic_network import DynamicNetwork


def calc_hepft_vol(dag: TaskDAG, network: NetworkGraph,
                   dynamic_net: DynamicNetwork,
                   volatility_threshold: float = 0.5,
                   availability_threshold: float = 0.7) -> dict:
    """
    HEPFT with volatile-processor filtering.

    volatility_threshold   — processors whose CV (std/mean) across snapshots
                             exceeds this are treated as unreliable and
                             excluded from the candidate set.
    availability_threshold — processors that were online for less than this
                             fraction of snapshots are also excluded.

    Both filters fall back to the full processor set if they would otherwise
    leave zero candidates.
    """

    # ── identify unreliable processors ────────────────────────────────────
    unreliable: set = set()
    for proc_id in network.processors:
        vol   = dynamic_net.proc_volatility(proc_id)
        avail = dynamic_net.proc_availability(proc_id)

        if vol > volatility_threshold:
            unreliable.add(proc_id)
        elif avail < availability_threshold:
            unreliable.add(proc_id)

    reliable_procs = [p for p in network.processors if p not in unreliable]
    if not reliable_procs:
        reliable_procs = list(network.processors.keys())

    # ── task prioritization ────────────────────────────────────────────────
    ranks = dag.compute_ranks(network, dynamic_network=dynamic_net)
    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

    # ── processor selection ────────────────────────────────────────────────
    schedule:       dict[int, tuple] = {}
    proc_available: dict[int, float] = {p: 0.0 for p in network.processors}

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]
        best_proc, best_est, best_eft = None, None, float('inf')

        # Primary pass — reliable processors only
        for proc_id in reliable_procs:
            ready_time = 0.0

            for parent_id in task.parents:
                parent_proc, _, parent_eft = schedule[parent_id]
                data_size = dag.edges[(parent_id, task_id)]

                # Use network state at the moment data transfer begins
                comm = dynamic_net.pred_net_func(parent_eft).comm_cost(
                    parent_proc, proc_id, data_size,
                    fallback_bandwidth=network.bandwidth
                )
                ready_time = max(ready_time, parent_eft + comm)

            est = max(ready_time, proc_available[proc_id])

            # Skip if processor is down at the planned start time
            if not dynamic_net.pred_net_func(est).has_processor(proc_id):
                continue

            eft = est + task.comp_costs[proc_id]

            if eft < best_eft:
                best_eft, best_est, best_proc = eft, est, proc_id

        # Fallback — all processors, static comm cost
        # (triggered only when every reliable processor is down at est)
        if best_proc is None:
            for proc_id in network.processors:
                ready_time = 0.0

                for parent_id in task.parents:
                    parent_proc, _, parent_eft = schedule[parent_id]
                    data_size = dag.edges[(parent_id, task_id)]
                    comm = network.comm_cost(parent_proc, proc_id, data_size)
                    ready_time = max(ready_time, parent_eft + comm)

                est = max(ready_time, proc_available[proc_id])
                eft = est + task.comp_costs[proc_id]

                if eft < best_eft:
                    best_eft, best_est, best_proc = eft, est, proc_id

        schedule[task_id]         = (best_proc, best_est, best_eft)
        proc_available[best_proc] = best_eft

    return schedule