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

def calc_hepft(dag: TaskDAG, network: NetworkGraph, dynamic_net: DynamicNetwork) -> tuple[dict, dict, dict]:

    # task prioritization
    ranks = dag.compute_ranks(network)
    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

     # processor selection
    proc_available_time = {proc_id: 0 for proc_id in network.processors}
    schedule = {} # task_id: processor, start_time, finish_time

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]
        best_proc, best_est, best_eft = None, 0, float('inf')
        processors = network.processor_list()
        
        for proc_id in processors:
            # inital list of processors, may need to account for available processors dynamicaly
            est = proc_available_time.get(proc_id, 0.0)
            predicted_network = dynamic_net.pred_net_func(est)
            processors = predicted_network.processor_list()

            for parent_id in task.parents:
                parent_proc, _, parent_finish = schedule[parent_id]

                transfer_time = max(parent_finish, est) # either the task starts once processor opens
                                                        # its queue or once parent finishes its task
                future_network = dynamic_net.pred_net_func(transfer_time)

                data_size = dag.edges[(parent_id, task_id)] # how much data the parent task is sending to current
                comm = future_network.comm_cost(parent_proc, proc_id, data_size, fallback_bandwidth=network.bandwidth)
                est = max(est, parent_finish + comm)

            # Check AFTER computing the actual start time
            net_at_start = dynamic_net.pred_net_func(est)
            if not net_at_start.has_processor(proc_id):
                continue

            # in case task was not assigned to processor
            if est == float('inf'):
                continue

            eft = est + task.comp_costs[proc_id]

            if eft < best_eft:
                best_eft = eft
                best_est = est
                best_proc = proc_id

        # After the processor loop, add a fallback to the base network
        if best_proc is None:
            for proc_id in network.processors:
                est = proc_available_time.get(proc_id, 0.0)
                for parent_id in task.parents:
                    parent_proc, _, parent_finish = schedule[parent_id]
                    data_size = dag.edges[(parent_id, task_id)]
                    comm = network.comm_cost(parent_proc, proc_id, data_size)
                    est = max(est, parent_finish + comm)
                eft = est + task.comp_costs[proc_id]
                if eft < best_eft:
                    best_eft, best_est, best_proc = eft, est, proc_id    

        schedule[task_id] = (best_proc, best_est, best_eft)
        proc_available_time[best_proc] = best_eft

    return schedule

def main():
    dag     = create_dag()
    network = create_network()
    dyn     = create_dynamic_network(network)

    # static HEFT
    static_schedule = calc_heft(dag, network)
    print("=== HEFT (static) ===")
    for task_id, (proc_id, start, finish) in sorted(static_schedule.items()):
        print(f"  Task {task_id} P{proc_id} | Start: {start:.2f} Finish: {finish:.2f}")

    # predictive HEPFT
    dynamic_schedule = calc_hepft(dag, network, dyn)
    print("\n=== HEPFT (dynamic) ===")
    for task_id, (proc_id, start, finish) in sorted(dynamic_schedule.items()):
        print(f"  Task {task_id} P{proc_id} | Start: {start:.2f} Finish: {finish:.2f}")


if __name__ == "__main__":
    main()