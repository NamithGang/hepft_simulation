class NetworkGraph:
    def __init__(self):
        self.processors = {}
        self.bandwidth = {}
        self.latency = {}

    def has_processor(self, processor):
        return processor in self.processors
  
    def processor_list(self):
        return self.processors
    
    def comm_cost(self, src_proc, dst_proc, data_size, fallback_bandwidth=None) -> float:
        if src_proc == dst_proc:
            return 0.0
        if (src_proc, dst_proc) not in self.bandwidth:
            if fallback_bandwidth and (src_proc, dst_proc) in fallback_bandwidth:
                return data_size / fallback_bandwidth[(src_proc, dst_proc)]
            return float('inf')  # link doesn't exist, treat as unreachable
        return (data_size / self.bandwidth[(src_proc, dst_proc)])

    def avg_comm_cost(self, data_size) -> float:
        avg_bw = sum(self.bandwidth.values()) / len(self.bandwidth)
        return data_size / avg_bw
    
    def comm_cost_integrated(self, src_proc, dst_proc, data_size, t_start,
                         dynamic_net, fallback_bandwidth=None) -> float:
        """Compute actual transfer time accounting for bandwidth changes."""
        if src_proc == dst_proc:
            return 0.0

        remaining = data_size
        t = t_start

        while remaining > 0:
            # get bandwidth at current time from the predicted network
            net = dynamic_net.pred_net_func(t)

            if (src_proc, dst_proc) in net.bandwidth:
                bw = net.bandwidth[(src_proc, dst_proc)]
            elif fallback_bandwidth and (src_proc, dst_proc) in fallback_bandwidth:
                bw = fallback_bandwidth[(src_proc, dst_proc)]
            else:
                return float('inf')  # link doesn't exist

            if bw <= 0:
                return float('inf')  # avoid division by zero

            # how long until the next snapshot changes bandwidth
            next_change = dynamic_net.next_snapshot_time(t)

            # time to finish transferring all remaining data at current bandwidth
            time_to_finish = remaining / bw

            if next_change == float('inf') or next_change >= t + time_to_finish:
                # no snapshot change before transfer completes — finish now
                t += time_to_finish
                remaining = 0
            else:
                # a snapshot change happens before transfer completes
                # transfer only what we can before the bandwidth changes
                interval    = next_change - t
                transferred = bw * interval
                remaining  -= transferred
                t           = next_change   # advance exactly to the next snapshot

        return t - t_start
    
class Processor:
    def __init__(self, proc_id, speed=1.0):
        self.id = proc_id
        self.speed = speed