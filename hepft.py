# hepft.py
"""
HEPFT scheduler: HEFT that plans around a predicted dynamic network.

calc_hepft() runs the steps below; the building blocks live in
hepft_helpers.py.

  1. Rank tasks on a time-averaged network            (rank_tasks)
  2. For each task, in rank order:
       a. compute when its input data would arrive on every processor, with
          communication costs that follow the changing bandwidth
       b. on each processor, find the earliest idle gap (including gaps
          between tasks already placed) where the task fits, the processor
          stays up for the whole run, and its output can still reach its
          children                                      (earliest_slot)
       c. place it on the processor where it finishes first
                                                        (choose_processor)
  3. If a task can never be placed, raise SchedulingError (no fallback).

Processor schedules are insertion-based, as in canonical HEFT.

The three dynamic parts can be switched on and off separately
(run_demo_toggle.py tests every combination):
  dynamic_ranking  step 1   time-averaged network   vs  starting network
  dynamic_comm     step 2a  comm_cost_integrated    vs  starting network's comm_cost
  dynamic_avail    step 2b  processors that join, stay up, output can leave
                            vs  starting processors, always up
"""
from __future__ import annotations

import bisect

from make_dag        import TaskDAG
from make_network    import NetworkGraph
from dynamic_network import DynamicNetwork
from hepft_helpers   import (
    INF,
    SchedulingError,          # re-exported: reactive.py imports it from here
    rank_tasks,
    all_processors,
    output_can_leave,
    choose_processor,
)


def calc_hepft(
    dag:         TaskDAG,
    network:     NetworkGraph,
    dynamic_net:     DynamicNetwork = None,
    is_dynamic:      bool           = True,
    dynamic_ranking: bool | None    = None,
    dynamic_comm:    bool | None    = None,
    dynamic_avail:   bool | None    = None,
) -> dict:
    """
    Schedule `dag` on `network`, planning around `dynamic_net`.

    is_dynamic=True   plan around the changing network in dynamic_net
    is_dynamic=False  static HEPFT: plan on the starting network only, the
                      same as (insertion-based) HEFT; dynamic_net not needed
    If is_dynamic=True but no dynamic_net is given, it also runs static.

    Each part can also be set on its own; any left as None follows is_dynamic:
      dynamic_ranking  rank tasks on the time-averaged network (True) or the
                       starting network (False)
      dynamic_comm     communication cost follows the changing bandwidth
                       (comm_cost_integrated) or uses the starting network
      dynamic_avail    use processors that join later, require the processor
                       to stay up and the output to be able to leave, waiting
                       for the next network change if needed (True); or treat
                       every starting processor as always up (False)
    A part set to True needs a dynamic_net (ValueError otherwise).

    Returns {task_id: (proc_id, start_time, finish_time)}.
    Raises SchedulingError if a task can never be placed; its .issues maps
    the task to the reason.
    """
    dynamic = is_dynamic and dynamic_net is not None
    dyn_rank  = dynamic if dynamic_ranking is None else dynamic_ranking
    dyn_comm  = dynamic if dynamic_comm    is None else dynamic_comm
    dyn_avail = dynamic if dynamic_avail   is None else dynamic_avail
    if dyn_comm and dynamic_net is None:
        raise ValueError("calc_hepft: dynamic_comm=True needs a dynamic_net")

    # ── 1. task order and candidate processors ─────────────────────────
    order      = rank_tasks(dag, network, dynamic_net, is_dynamic=dyn_rank)
    candidates = all_processors(network, dynamic_net, is_dynamic=dyn_avail)
    schedule   = {}                                # task_id -> (proc, start, finish)
    busy       = {p: [] for p in candidates}       # proc -> sorted (start, finish) placed

    def data_ready(task_id: int, proc_id: int) -> float:
        """When all of task_id's input data would have arrived on proc_id."""
        ready = 0.0
        for parent_id in dag.nodes[task_id].parents:
            parent_proc, _, parent_eft = schedule[parent_id]
            data_size = dag.edges[(parent_id, task_id)]
            if dyn_comm:
                comm = dynamic_net.comm_cost_integrated(
                    parent_proc, proc_id, data_size, parent_eft)
            else:
                comm = network.comm_cost(parent_proc, proc_id, data_size)
            ready = max(ready, parent_eft + comm)
        return ready

    # ── 2. place each task in rank order ───────────────────────────────
    for task_id in order:
        # a. data-ready time and run time on every processor the data can reach
        options = {}
        for proc_id in candidates:
            if proc_id not in dag.nodes[task_id].comp_costs:
                continue                           # no run time for this processor
            ready = data_ready(task_id, proc_id)
            if ready != INF:                       # inf = data can never arrive
                options[proc_id] = (ready, dag.nodes[task_id].comp_costs[proc_id])

        # b + c. earliest usable gap on each processor, then the earliest finish
        #        (static mode: any gap that fits, as in HEFT)
        choice = choose_processor(
            options,
            busy,
            lambda p, finish: output_can_leave(
                dag, task_id, p, finish, candidates, dynamic_net, is_dynamic=dyn_avail),
            dynamic_net,
            is_dynamic=dyn_avail)

        # 3. no fallback: a task that can never be placed is an error
        if choice is None:
            reason = ("its input data can't reach any processor" if not options
                      else "no reachable processor is ever up long enough to run it")
            raise SchedulingError(f"Task {task_id}: {reason}", {task_id: reason})

        best_proc, best_est, best_eft = choice
        schedule[task_id] = (best_proc, best_est, best_eft)
        bisect.insort(busy[best_proc], (best_est, best_eft))   # keep sorted by start

    return schedule