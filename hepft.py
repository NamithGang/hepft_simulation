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
    dynamic_net: DynamicNetwork = None,
    is_dynamic:  bool           = True,
) -> dict:
    """
    Schedule `dag` on `network`, planning around `dynamic_net`.

    is_dynamic=True   plan around the changing network in dynamic_net
    is_dynamic=False  static HEPFT: plan on the starting network only, the
                      same as (insertion-based) HEFT; dynamic_net not needed
    If is_dynamic=True but no dynamic_net is given, it also runs static.
    The flag is passed to every helper in hepft_helpers.py.

    Returns {task_id: (proc_id, start_time, finish_time)}.
    Raises SchedulingError if a task can never be placed; its .issues maps
    the task to the reason.
    """
    dynamic = is_dynamic and dynamic_net is not None

    # ── 1. task order and candidate processors ─────────────────────────
    order      = rank_tasks(dag, network, dynamic_net, is_dynamic=dynamic)
    candidates = all_processors(network, dynamic_net, is_dynamic=dynamic)
    schedule   = {}                                # task_id -> (proc, start, finish)
    busy       = {p: [] for p in candidates}       # proc -> sorted (start, finish) placed

    def data_ready(task_id: int, proc_id: int) -> float:
        """When all of task_id's input data would have arrived on proc_id."""
        ready = 0.0
        for parent_id in dag.nodes[task_id].parents:
            parent_proc, _, parent_eft = schedule[parent_id]
            data_size = dag.edges[(parent_id, task_id)]
            if dynamic:
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
                dag, task_id, p, finish, candidates, dynamic_net, is_dynamic=dynamic),
            dynamic_net,
            is_dynamic=dynamic)

        # 3. no fallback: a task that can never be placed is an error
        if choice is None:
            reason = ("its input data can't reach any processor" if not options
                      else "no reachable processor is ever up long enough to run it")
            raise SchedulingError(f"Task {task_id}: {reason}", {task_id: reason})

        best_proc, best_est, best_eft = choice
        schedule[task_id] = (best_proc, best_est, best_eft)
        bisect.insort(busy[best_proc], (best_est, best_eft))   # keep sorted by start

    return schedule