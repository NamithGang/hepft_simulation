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
        link_values: dict[tuple, list[float]] = {}

        for _, net in self.snapshots:
            for (src, dst), bw in net.bandwidth.items():
                if src == proc_id or dst == proc_id:
                    key = (src, dst)
                    if key not in link_values:
                        link_values[key] = []
                    link_values[key].append(bw)

        if not link_values:
            return float('inf')

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
            cv_scores.append(std / mean)

        return sum(cv_scores) / len(cv_scores)

    def next_snapshot_time(self, t: float) -> float:
        idx = bisect.bisect_right(self.timestamps, t)
        if idx >= len(self.snapshots):
            return float('inf')
        return self.snapshots[idx][0]

    def proc_availability(self, proc_id: int) -> dict[float, float]:
        """
        Returns a dict of {up_time: down_time} representing every interval
        during which proc_id was online.

        key   = timestamp when the processor came back up (or first appeared)
        value = timestamp when the processor next went down (or the horizon
                if it never went down again)

        Example:
            {0.0: 28.3, 41.7: 95.2}
            means the processor was up from t=0   to t=28.3
            then  up again      from t=41.7 to t=95.2

        An empty dict means the processor never appeared in any snapshot.
        """
        if not self.snapshots:
            return {}

        intervals: dict[float, float] = {}
        up_since: float | None = None

        for ts, net in self.snapshots:
            is_up = net.has_processor(proc_id)

            if is_up and up_since is None:
                # processor just came online — open a new interval
                up_since = ts

            elif not is_up and up_since is not None:
                # processor just went offline — close the interval
                intervals[up_since] = ts
                up_since = None

        # if still up at the last known snapshot, close against the horizon
        if up_since is not None:
            intervals[up_since] = self.timestamps[-1]

        return intervals

    def proc_up_at_time(self, proc_id: int, t: float) -> bool:
        """
        Query whether proc_id is up at time t using the availability intervals.
        Returns False if the processor never appeared or t is outside all intervals.
        """
        intervals = self.proc_availability(proc_id)
        return any(up <= t < down for up, down in intervals.items())

    def next_online_time(self, proc_id: int, t: float) -> float:
        """
        When does proc_id next become available after time t?
        Returns float('inf') if the processor never comes back within the horizon.
        """
        intervals = self.proc_availability(proc_id)
        future = [up for up in intervals if up > t]
        if not future:
            return float('inf')
        return min(future)

    def comm_cost_integrated(self, src_proc, dst_proc, data_size, t_start) -> float:
        """Compute actual transfer time accounting for bandwidth changes."""
        if src_proc == dst_proc:
            return 0.0

        remaining = data_size
        t = t_start

        while remaining > 1e-9:
            net = self.pred_net_func(t)

            if (src_proc, dst_proc) not in net.bandwidth:
                return float('inf')

            bw = net.bandwidth[(src_proc, dst_proc)]
            if bw <= 0:
                return float('inf')

            time_to_finish = remaining / bw
            next_change    = self.next_snapshot_time(t)

            if next_change == float('inf') or next_change >= t + time_to_finish:
                t        += time_to_finish
                remaining = 0.0
            else:
                interval   = next_change - t
                remaining -= bw * interval
                t          = next_change

        return t - t_start