# dynamic_network.py
import bisect
from typing import Optional
from make_network import NetworkGraph

class DynamicNetwork:
    def __init__(self, base_network: NetworkGraph):
        self.base_network = base_network
        self.snapshots = []
        self.timestamps = []
        self.add_snapshot(0.0, base_network)

    def add_snapshot(self, timestamp: float, network: NetworkGraph):
        self.snapshots.append((timestamp, network))
        self.snapshots.sort(key=lambda x: x[0])
        self.timestamps = [s[0] for s in self.snapshots]

    def pred_net_func(self, t: float) -> NetworkGraph:
        if not self.snapshots:
            return self.base_network
        if t <= self.timestamps[0]:
            return self.snapshots[0][1]
        if t >= self.timestamps[-1]:
            return self.snapshots[-1][1]
        idx = bisect.bisect_right(self.timestamps, t) - 1
        return self.snapshots[idx][1]

    def proc_volatility(self, proc_id: int) -> float:
        """
        Measure how volatile a processor's links are across all snapshots.
        Returns a score between 0.0 (perfectly stable) and higher values
        meaning more volatile.

        Method: for each link involving this processor, compute the
        coefficient of variation (std / mean) of bandwidth across snapshots.
        Average that across all links.
        """
        # collect bandwidth values per link involving this processor
        link_values: dict[tuple, list[float]] = {}

        for _, net in self.snapshots:
            for (src, dst), bw in net.bandwidth.items():
                if src == proc_id or dst == proc_id:
                    key = (src, dst)
                    if key not in link_values:
                        link_values[key] = []
                    link_values[key].append(bw)

        if not link_values:
            return float('inf')  # never appeared in any snapshot = maximally unreliable

        cv_scores = []
        for values in link_values.values():
            if len(values) < 2:
                cv_scores.append(0.0)
                continue
            mean = sum(values) / len(values)
            if mean == 0:
                cv_scores.append(float('inf'))
                continue
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            cv_scores.append(std / mean)   # coefficient of variation

        return sum(cv_scores) / len(cv_scores)
    def next_snapshot_time(self, t: float) -> float:
        """
        Returns the timestamp of the next snapshot strictly after time t.
        Returns float('inf') if there are no snapshots after t.
        """
        idx = bisect.bisect_right(self.timestamps, t)
        if idx >= len(self.snapshots):
            return float('inf')
        return self.snapshots[idx][0]

    def proc_availability(self, proc_id: int) -> float:
        """
        Returns the fraction of snapshots in which this processor was present.
        1.0 = always available, 0.0 = never available.
        """
        if not self.snapshots:
            return 0.0
        present = sum(
            1 for _, net in self.snapshots
            if proc_id in net.processors
        )
        return present / len(self.snapshots)