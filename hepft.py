from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from demo import create_dag, create_network, create_dynamic_network
from dynamic_network import DynamicNetwork

def calc_hepft(dag: TaskDAG, network: NetworkGraph,
               dynamic_net: DynamicNetwork) -> dict:

    # task prioritization
    ranks = dag.compute_ranks(network, dynamic_network=dynamic_net)
    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

    # processor selection
    schedule       = {}
    proc_available = {proc_id: 0.0 for proc_id in network.processors}

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]
        best_proc, best_est, best_eft = None, None, float('inf')

        for proc_id in network.processors:

            # ── compute EST ───────────────────────────────────────────────
            ready_time = 0.0
            for parent_id in task.parents:
                parent_proc, _, parent_eft = schedule[parent_id]
                data_size = dag.edges[(parent_id, task_id)]
                comm = dynamic_net.pred_net_func(parent_eft).comm_cost_integrated(
                    parent_proc, proc_id, data_size, parent_eft, dynamic_net,
                    fallback_bandwidth=network.bandwidth
                )
                ready_time = max(ready_time, parent_eft + comm)

            est = max(ready_time, proc_available[proc_id])
            eft = est + task.comp_costs[proc_id]

            # ── full window check [est, eft] ───────────────────────────────
            # with perfect knowledge, scan every snapshot inside the
            # execution window. if the processor is absent from any of them
            # it will fail mid-execution — skip it for this task only.
            proc_goes_down = False

            # check boundaries first
            if not dynamic_net.pred_net_func(est).has_processor(proc_id):
                proc_goes_down = True
            elif not dynamic_net.pred_net_func(eft).has_processor(proc_id):
                proc_goes_down = True
            else:
                # walk every snapshot strictly inside (est, eft)
                t_cursor = est
                while True:
                    next_t = dynamic_net.next_snapshot_time(t_cursor)
                    if next_t == float('inf') or next_t >= eft:
                        break
                    snapshot = dynamic_net.pred_net_func(next_t)
                    if not snapshot.has_processor(proc_id):
                        proc_goes_down = True
                        break
                    t_cursor = next_t

            if proc_goes_down:
                continue   # skip for this task only — reconsidered next task

            # ── update best ───────────────────────────────────────────────
            if eft < best_eft:
                best_eft, best_est, best_proc = eft, est, proc_id

        # ── fallback: every processor fails during this window ─────────────
        if best_proc is None:
            for proc_id in network.processors:
                ready_time = 0.0
                for parent_id in task.parents:
                    parent_proc, _, parent_eft = schedule[parent_id]
                    data_size = dag.edges[(parent_id, task_id)]
                    comm      = network.comm_cost(parent_proc, proc_id, data_size)
                    ready_time = max(ready_time, parent_eft + comm)

                est = max(ready_time, proc_available[proc_id])
                eft = est + task.comp_costs[proc_id]

                if eft < best_eft:
                    best_eft, best_est, best_proc = eft, est, proc_id

        schedule[task_id]         = (best_proc, best_est, best_eft)
        proc_available[best_proc] = best_eft

    return schedule