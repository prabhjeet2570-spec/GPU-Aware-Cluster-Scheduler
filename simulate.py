#!/usr/bin/env python3
from __future__ import annotations

import argparse

from cluster import Cluster
from job import JobGenerator
from schedulers import (
    NaiveScheduler,
    BinPackScheduler,
    SpreadScheduler,
    GangScheduler,
)
from simulation import Simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-Aware Cluster Scheduler Simulation"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="naive",
        choices=["naive", "bin-pack", "spread", "gang"],
        help="Scheduling algorithm to use",
    )
    parser.add_argument(
        "--jobs", type=int, default=500, help="Number of jobs to simulate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for job generation"
    )
    parser.add_argument(
        "--nodes", type=int, default=3, help="Number of nodes in the cluster"
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="GPUs per node",
    )
    parser.add_argument(
        "--memory-per-gpu",
        type=int,
        default=80,
        help="Memory per GPU (GB)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=10_000,
        help="Maximum simulation ticks",
    )
    parser.add_argument(
        "--preempt",
        action="store_true",
        help="Enable preemption for high-priority jobs",
    )
    return parser.parse_args()


def build_scheduler(name: str):
    schedulers = {
        "naive": NaiveScheduler(),
        "bin-pack": BinPackScheduler(),
        "spread": SpreadScheduler(),
        "gang": GangScheduler(),
    }
    return schedulers[name]


def main() -> None:
    args = parse_args()

    cluster = Cluster.create_uniform(
        node_count=args.nodes,
        gpus_per_node=args.gpus_per_node,
        memory_per_gpu=args.memory_per_gpu,
    )

    generator = JobGenerator(seed=args.seed)
    jobs = generator.generate(
        count=args.jobs,
        arrival_window=max(args.jobs // 5, 10),
    )

    scheduler = build_scheduler(args.scheduler)

    sim = Simulation(
        cluster=cluster,
        scheduler=scheduler,
        jobs=jobs,
        max_ticks=args.max_ticks,
        enable_preemption=args.preempt,
    )

    result = sim.run()

    print(f"=== Simulation Results (scheduler={args.scheduler}) ===")
    print(f"  Total jobs:              {result.total_jobs}")
    print(f"  Completed:               {result.completed_jobs}")
    print(f"  Preempted:               {result.preempted_jobs}")
    print(f"  Cross-node jobs:         {result.cross_node_jobs}")
    print(f"  Total ticks:             {result.total_ticks}")
    print(f"  Avg wait time:           {result.avg_wait_time:.2f} ticks")
    print(f"  Max wait time:           {result.max_wait_time}")
    print(f"  Avg completion time:     {result.avg_completion_time:.2f} ticks")
    print(f"  Avg fragmentation:       {result.avg_fragmentation:.2%}")
    print(f"  Peak fragmentation:      {result.peak_fragmentation:.2%}")
    print(f"  Avg node util std dev:   {result.avg_node_std:.3f}")
    print(f"  Peak node util std dev:  {result.peak_node_std:.3f}")
    print(f"  Jobs/tick:               {result.jobs_scheduled_per_tick:.3f}")
    print(f"  Schedule attempts:       {result.schedule_attempts}")
    print(f"  Schedule failures:       {result.schedule_failures}")


if __name__ == "__main__":
    main()
