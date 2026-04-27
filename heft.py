from make_dag import TaskDAG, Task
from make_network import NetworkGraph, Processor
from demo import create_dag, create_network, create_dynamic_network
from dynamic_network import DynamicNetwork

def calc_heft(dag: TaskDAG, network: NetworkGraph) -> dict:
    
    # task prioritization
    ranks = dag.compute_ranks(network)
    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)
    
    # processor selection
    schedule = {} # task_id: processor, start_time, finish_time
    proc_available = {proc_id: 0 for proc_id in network.processors}
    
    for task_id in sorted_tasks:
        task = dag.nodes[task_id]
        best_proc, best_est, best_eft = None, None, float('inf')
        
        for proc_id in network.processors:
            ready_time = 0.0
            for parent_id in task.parents:
                parent_proc, _, parent_eft = schedule[parent_id]
                data_size = dag.edges[(parent_id, task_id)] # how much data the parent task is sending to current
                comm = network.comm_cost(parent_proc, proc_id, data_size)
                ready_time = max(ready_time, parent_eft + comm)
            
            est = max(ready_time, proc_available[proc_id])
            eft = est + task.comp_costs[proc_id]
            
            if eft < best_eft:
                best_eft, best_est, best_proc = eft, est, proc_id
        
        schedule[task_id] = (best_proc, best_est, best_eft)
        proc_available[best_proc] = best_eft
    
    return schedule