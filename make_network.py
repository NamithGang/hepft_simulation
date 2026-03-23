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
        return data_size / self.bandwidth[(src_proc, dst_proc)]

    def avg_comm_cost(self, data_size) -> float:
        avg_bw = sum(self.bandwidth.values()) / len(self.bandwidth)
        return data_size / avg_bw
    
class Processor:
    def __init__(self, proc_id, speed=1.0):
        self.id = proc_id
        self.speed = speed