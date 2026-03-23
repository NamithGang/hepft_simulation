# dynamic_network.py
import bisect
from make_network import NetworkGraph

class DynamicNetwork:
    """
    Stores a sequence of NetworkGraph snapshots at known timestamps.
    pred_net_func(t) returns the most recent snapshot at or before time t.
    """

    def __init__(self, base_network: NetworkGraph):
        self.base_network = base_network
        self.snapshots = []        # list of (timestamp, NetworkGraph), kept sorted
        self.timestamps = []       # kept in sync with snapshots for bisect lookup
        self.add_snapshot(0.0, base_network)

    def add_snapshot(self, timestamp: float, network: NetworkGraph):
        """Register a known network state at a given time."""
        self.snapshots.append((timestamp, network))
        self.snapshots.sort(key=lambda x: x[0])
        self.timestamps = [s[0] for s in self.snapshots]

    def pred_net_func(self, t: float) -> NetworkGraph:
        """
        Return the most recent network snapshot at or before time t.
        This directly reflects what the test case generated at that time.
        """
        if not self.snapshots:
            return self.base_network

        # before first snapshot
        if t <= self.timestamps[0]:
            return self.snapshots[0][1]

        # after last snapshot
        if t >= self.timestamps[-1]:
            return self.snapshots[-1][1]

        # find the most recent snapshot at or before t
        idx = bisect.bisect_right(self.timestamps, t) - 1
        return self.snapshots[idx][1]