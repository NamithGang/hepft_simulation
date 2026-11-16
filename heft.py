# heft.py
"""
HEFT (Heterogeneous Earliest Finish Time) on a fixed network.

  1. Task prioritization: order tasks by upward rank, highest first.
  2. Processor selection: give each task, in that order, to the processor
     where it finishes earliest.

Processor schedules are INSERTION-BASED, as in canonical HEFT (Topcuoglu et
al., 2002): a task can go into an idle gap between tasks already placed on a
processor, as long as the whole run fits, not only after the last task.
"""
from __future__ import annotations

import bisect

from make_dag     import TaskDAG
from make_network import NetworkGraph


def _fit_in_gap(busy: list[tuple[float, float]], ready: float, duration: float) -> float:
    """
    Earliest start >= `ready` at which a task of length `duration` fits
    without overlapping any (start, finish) interval in `busy` (sorted).
    """
    start = ready
    for b_start, b_finish in busy:
        if start + duration <= b_start:           # fits in the gap before this task
            return start
        start = max(start, b_finish)              # otherwise try after it
    return start


def calc_heft(dag: TaskDAG, network: NetworkGraph) -> dict:
    """Returns {task_id: (proc_id, start_time, finish_time)}."""

    # ── 1. task prioritization ─────────────────────────────────────────
    ranks        = dag.compute_ranks(network)
    sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

    # ── 2. processor selection ─────────────────────────────────────────
    schedule = {}                                      # task_id -> (proc, start, finish)
    busy     = {proc_id: [] for proc_id in network.processors}   # sorted (start, finish)

    for task_id in sorted_tasks:
        task = dag.nodes[task_id]
        best_proc, best_est, best_eft = None, None, float('inf')

        for proc_id in network.processors:
            # when all input data would have arrived on proc_id
            ready_time = 0.0
            for parent_id in task.parents:
                parent_proc, _, parent_eft = schedule[parent_id]
                data_size = dag.edges[(parent_id, task_id)]   # data sent parent -> task
                comm = network.comm_cost(parent_proc, proc_id, data_size)
                ready_time = max(ready_time, parent_eft + comm)

            # earliest idle gap on proc_id that fits the whole run
            est = _fit_in_gap(busy[proc_id], ready_time, task.comp_costs[proc_id])
            eft = est + task.comp_costs[proc_id]

            if eft < best_eft:
                best_eft, best_est, best_proc = eft, est, proc_id

        schedule[task_id] = (best_proc, best_est, best_eft)
        bisect.insort(busy[best_proc], (best_est, best_eft))

    return schedule