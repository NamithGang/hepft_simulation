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
        best_proc, best_est, best_eft = None, 0.0, float('inf')

        # Always consider ALL processors
        for proc_id in network.processors:
            base_est = proc_available_time.get(proc_id, 0.0)
            est = base_est

            for parent_id in task.parents:
                parent_proc, _, parent_finish = schedule[parent_id]

                transfer_time = max(parent_finish, est)
                future_network = dynamic_net.pred_net_func(transfer_time)

                data_size = dag.edges[(parent_id, task_id)]
                comm = future_network.comm_cost(
                    parent_proc,
                    proc_id,
                    data_size,
                    fallback_bandwidth=network.bandwidth
                )

                est = max(est, parent_finish + comm)

            # Check processor availability at actual start time
            net_at_start = dynamic_net.pred_net_func(est)
            if not net_at_start.has_processor(proc_id):
                continue

            eft = est + task.comp_costs[proc_id]

            if eft < best_eft:
                best_eft = eft
                best_est = est
                best_proc = proc_id

        # Fallback (unchanged, but now rarely triggered)
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

def simulate_on_dynamic(dag, dynamic_net, schedule):
    # schedule task_id: processor, start_time, finish_time
    ordered = sorted(schedule.items(), key=lambda item: item[1][1]) # sorted by start time

    actual = {} # task_id: processor, start_time, finish_time
    proc_available = {pid: 0.0 for pid in dynamic_net.pred_net_func(0.0).processors}

    for task_id, (proc_id, start_time, finish_time) in ordered:
        ready_time = proc_available.get(proc_id, 0.0)

        for parent_id in dag.nodes[task_id].parents:
            parent_proc, _, parent_finish = actual[parent_id]
            net_at_t = dynamic_net.pred_net_func(parent_finish)  # checking state of network at the end of each task completion
            data_size = dag.edges[(parent_id, task_id)] # how much data the parent task is sending
            comm = net_at_t.comm_cost(
                parent_proc, proc_id, data_size,
                fallback_bandwidth=dynamic_net.base_network.bandwidth
            )
            ready_time = max(ready_time, parent_finish + comm) # when task actually starts

        start = ready_time
        finish = start + dag.nodes[task_id].comp_costs[proc_id]
        actual[task_id] = (proc_id, start, finish)
        proc_available[proc_id] = finish

    return actual

def main():
    from demo import get_test_cases
    test_cases = get_test_cases()

    for tc in test_cases:
        print(f"\n{'='*55}")
        print(f" {tc['name']}")
        print(f"{'='*55}")

        dag     = tc['dag']
        network = tc['network']
        dyn     = tc['dynamic_network']

        # planned schedules
        static_schedule  = calc_heft(dag, network)
        dynamic_schedule = calc_hepft(dag, network, dyn)

        # planned HEFT
        static_start, static_finish = float('inf'), 0
        print("\n=== HEFT (planned) ===")
        for task_id, (proc_id, start, finish) in sorted(static_schedule.items()):
            if static_start > start:
                static_start = start
            if static_finish < finish:
                static_finish = finish
        planned_time_heft = static_finish - static_start
        print(f"\t Expected HEFT execution time: {planned_time_heft:.2f}")

        # planned HEPFT
        dynamic_start, dynamic_finish = float('inf'), 0
        print("\n=== HEPFT (planned) ===")
        for task_id, (proc_id, start, finish) in sorted(dynamic_schedule.items()):
            if dynamic_start > start:
                dynamic_start = start
            if dynamic_finish < finish:
                dynamic_finish = finish
        planned_time_hepft = dynamic_finish - dynamic_start
        print(f"\t Expected HEPFT execution time: {planned_time_hepft:.2f}")

        # simulated HEFT
        true_static_schedule = simulate_on_dynamic(dag, dyn, static_schedule)
        static_start, static_finish = float('inf'), 0
        print("\n=== HEFT (simulated) ===")
        for task_id, (proc_id, start, finish) in sorted(true_static_schedule.items()):
            if static_start > start:
                static_start = start
            if static_finish < finish:
                static_finish = finish
        time_heft = static_finish - static_start
        print(f"\t Simulated HEFT execution time: {time_heft:.2f}")

        # simulated HEPFT
        true_dynamic_schedule = simulate_on_dynamic(dag, dyn, dynamic_schedule)
        dynamic_start, dynamic_finish = float('inf'), 0
        print("\n=== HEPFT (simulated) ===")
        for task_id, (proc_id, start, finish) in sorted(true_dynamic_schedule.items()):
            if dynamic_start > start:
                dynamic_start = start
            if dynamic_finish < finish:
                dynamic_finish = finish
        time_hepft = dynamic_finish - dynamic_start
        print(f"\t Simulated HEPFT execution time: {time_hepft:.2f}")

        # winner
        if time_hepft > time_heft:
            dif = time_hepft - time_heft
            print(f"\nHEFT wins by {dif:.2f} ({(dif/time_heft)*100:.2f}%)")
        else:
            dif = time_heft - time_hepft
            print(f"\nHEPFT wins by {dif:.2f} ({(dif/time_hepft)*100:.2f}%)")

if __name__ == "__main__":
    main()