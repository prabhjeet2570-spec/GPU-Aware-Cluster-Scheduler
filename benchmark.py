#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from cluster import Cluster
from job import JobGenerator
from schedulers import (
    NaiveScheduler,
    BinPackScheduler,
    SpreadScheduler,
    GangScheduler,
)
from simulation import Simulation


@dataclass
class SchedulerRun:
    avg_wait_time: float = 0.0
    max_wait_time: int = 0
    avg_completion_time: float = 0.0
    final_fragmentation: float = 0.0
    peak_fragmentation: float = 0.0
    avg_fragmentation: float = 0.0
    final_node_std: float = 0.0
    avg_node_std: float = 0.0
    peak_node_std: float = 0.0
    cross_node_jobs: int = 0
    jobs_per_tick: float = 0.0
    completed: int = 0
    preempted: int = 0
    total_ticks: int = 0
    schedule_failures: int = 0


@dataclass
class SchedulerStats:
    name: str
    runs: List[SchedulerRun] = field(default_factory=list)

    def add(self, run: SchedulerRun) -> None:
        self.runs.append(run)

    def _vals(self, attr: str) -> List[float]:
        return [getattr(r, attr) for r in self.runs]

    def mean(self, attr: str) -> float:
        return mean(self._vals(attr))

    def std(self, attr: str) -> float:
        if len(self.runs) < 2:
            return 0.0
        return stdev(self._vals(attr))

    def summary(self) -> Dict[str, Tuple[float, float]]:
        return {
            "avg_wait_time": (self.mean("avg_wait_time"), self.std("avg_wait_time")),
            "max_wait_time": (self.mean("max_wait_time"), self.std("max_wait_time")),
            "avg_completion_time": (
                self.mean("avg_completion_time"),
                self.std("avg_completion_time"),
            ),
            "final_fragmentation": (
                self.mean("final_fragmentation"),
                self.std("final_fragmentation"),
            ),
            "avg_fragmentation": (
                self.mean("avg_fragmentation"),
                self.std("avg_fragmentation"),
            ),
            "peak_fragmentation": (
                self.mean("peak_fragmentation"),
                self.std("peak_fragmentation"),
            ),
            "avg_node_std": (self.mean("avg_node_std"), self.std("avg_node_std")),
            "peak_node_std": (self.mean("peak_node_std"), self.std("peak_node_std")),
            "cross_node_jobs": (
                self.mean("cross_node_jobs"),
                self.std("cross_node_jobs"),
            ),
            "jobs_per_tick": (self.mean("jobs_per_tick"), self.std("jobs_per_tick")),
            "completed": (self.mean("completed"), self.std("completed")),
            "preempted": (self.mean("preempted"), self.std("preempted")),
        }


SCHEDULERS: Dict[str, type] = {
    "naive": NaiveScheduler,
    "bin-pack": BinPackScheduler,
    "spread": SpreadScheduler,
    "gang": GangScheduler,
}


def run_scheduler(
    scheduler_name: str,
    jobs_count: int,
    cluster_cfg: Tuple[int, int, int],
    seed: int,
    preempt: bool = False,
) -> SchedulerRun:
    cluster = Cluster.create_uniform(*cluster_cfg)
    generator = JobGenerator(seed=seed)
    jobs = generator.generate(
        count=jobs_count,
        arrival_window=max(jobs_count // 5, 10),
    )
    scheduler = SCHEDULERS[scheduler_name]()
    sim = Simulation(
        cluster=cluster,
        scheduler=scheduler,
        jobs=jobs,
        enable_preemption=preempt,
    )
    result = sim.run()

    return SchedulerRun(
        avg_wait_time=result.avg_wait_time,
        max_wait_time=result.max_wait_time,
        avg_completion_time=result.avg_completion_time,
        final_fragmentation=result.final_fragmentation,
        peak_fragmentation=result.peak_fragmentation,
        avg_fragmentation=result.avg_fragmentation,
        final_node_std=result.final_node_std,
        avg_node_std=result.avg_node_std,
        peak_node_std=result.peak_node_std,
        cross_node_jobs=result.cross_node_jobs,
        jobs_per_tick=result.jobs_scheduled_per_tick,
        completed=result.completed_jobs,
        preempted=result.preempted_jobs,
        total_ticks=result.total_ticks,
        schedule_failures=result.schedule_failures,
    )


def run_benchmark(
    scheduler_names: List[str],
    jobs_count: int,
    cluster_cfg: Tuple[int, int, int],
    seeds: List[int],
    preempt: bool = False,
) -> Dict[str, SchedulerStats]:
    results: Dict[str, SchedulerStats] = {
        name: SchedulerStats(name=name) for name in scheduler_names
    }

    total_runs = len(scheduler_names) * len(seeds)
    run_num = 0

    for seed in seeds:
        for sched_name in scheduler_names:
            run_num += 1
            print(
                f"  [{run_num}/{total_runs}] {sched_name:>8s} seed={seed}",
                end="",
                flush=True,
            )
            run = run_scheduler(sched_name, jobs_count, cluster_cfg, seed, preempt)
            results[sched_name].add(run)
            print(
                f"  wait={run.avg_wait_time:.0f}"
                f" frag={run.avg_fragmentation:.0%}"
                f" cross={run.cross_node_jobs}"
            )

    return results


def print_comparison_table(
    results: Dict[str, SchedulerStats],
    baseline: str = "naive",
    preempt: bool = False,
) -> None:
    baseline_stats = results[baseline]

    print(f"\n{'=' * 100}")
    title = f"Benchmark Comparison ({'with preemption' if preempt else 'no preemption'})"
    print(f"{title:^100}")
    print(f"{'=' * 100}")

    metrics = [
        ("avg_wait_time", "Avg Wait Time (ticks)", "{:.1f}"),
        ("max_wait_time", "Max Wait Time (ticks)", "{:.0f}"),
        ("avg_completion_time", "Avg Completion Time (ticks)", "{:.1f}"),
        ("avg_fragmentation", "Avg Fragmentation", "{:.2%}"),
        ("peak_fragmentation", "Peak Fragmentation", "{:.2%}"),
        ("avg_node_std", "Node Utilization Std Dev", "{:.3f}"),
        ("cross_node_jobs", "Cross-Node Jobs", "{:.0f}"),
        ("jobs_per_tick", "Jobs / Tick", "{:.4f}"),
        ("completed", "Jobs Completed", "{:.0f}"),
    ]

    header = f"{'Metric':<30}"
    for name in results:
        header += f"  {name:>10}"
    header += f"  {'vs naive':>10}"
    print(header)
    print("-" * 100)

    for attr, label, fmt in metrics:
        row = f"{label:<30}"
        baseline_mean = baseline_stats.mean(attr)

        for name in results:
            stats = results[name]
            m = stats.mean(attr)
            s = stats.std(attr)
            formatted = fmt.format(m)
            formatted_s = fmt.format(s)
            row += f"  {formatted:>8} ±{formatted_s:>6}"

        row += f"  {'':>10}"
        print(row)

    print("-" * 100)

    improvement_row = f"{'Wait time improvement':<30}"
    for name in results:
        if name == baseline:
            improvement_row += f"  {'---':>10}"
            continue
        imp = (
            1 - results[name].mean("avg_wait_time") / baseline_stats.mean("avg_wait_time")
        )
        improvement_row += f"  {imp:>+9.1%}"
    improvement_row += f"  {'vs naive':>10}"
    print(improvement_row)

    print("=" * 100)


def plot_comparison(
    results: Dict[str, SchedulerStats],
    output_dir: str = ".",
    preempt: bool = False,
) -> None:
    if not HAS_MPL:
        print("matplotlib not available, skipping charts")
        return

    names = list(results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Scheduler Benchmark Comparison{' (with preemption)' if preempt else ''}",
        fontsize=14,
    )

    plot_configs = [
        ("avg_wait_time", "Avg Wait Time (ticks)", axes[0, 0]),
        ("avg_fragmentation", "Avg Fragmentation", axes[0, 1]),
        ("avg_node_std", "Node Util Std Dev", axes[0, 2]),
        ("cross_node_jobs", "Cross-Node Jobs", axes[1, 0]),
        ("jobs_per_tick", "Jobs / Tick", axes[1, 1]),
        ("completed", "Jobs Completed", axes[1, 2]),
    ]

    for attr, ylabel, ax in plot_configs:
        means = [results[n].mean(attr) for n in names]
        stds = [results[n].std(attr) for n in names]

        bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors[: len(names)])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)

        for bar, val in zip(bars, means):
            if attr in ("avg_fragmentation", "peak_fragmentation"):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            elif attr == "jobs_per_tick":
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU Cluster Scheduler Benchmark"
    )
    parser.add_argument(
        "--jobs", type=int, default=500, help="Jobs per run"
    )
    parser.add_argument(
        "--nodes", type=int, default=5, help="Number of nodes"
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=8, help="GPUs per node"
    )
    parser.add_argument(
        "--memory-per-gpu", type=int, default=80, help="Memory per GPU (GB)"
    )
    parser.add_argument(
        "--seeds", type=int, default=20, help="Number of random seeds"
    )
    parser.add_argument(
        "--start-seed", type=int, default=1, help="Starting seed value"
    )
    parser.add_argument(
        "--preempt", action="store_true", help="Enable preemption"
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        nargs="+",
        default=list(SCHEDULERS.keys()),
        choices=list(SCHEDULERS.keys()),
        help="Schedulers to benchmark",
    )
    parser.add_argument(
        "--output", type=str, default=".", help="Output directory for charts"
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart generation"
    )
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    cluster_cfg = (args.nodes, args.gpus_per_node, args.memory_per_gpu)

    print(f"Benchmark: {args.jobs} jobs, {args.seeds} seeds")
    print(f"Cluster:  {args.nodes} nodes x {args.gpus_per_node} GPUs")
    print(f"Schedulers: {', '.join(args.schedulers)}")
    print(f"Preemption: {'enabled' if args.preempt else 'disabled'}")
    print()

    results = run_benchmark(
        scheduler_names=args.schedulers,
        jobs_count=args.jobs,
        cluster_cfg=cluster_cfg,
        seeds=seeds,
        preempt=args.preempt,
    )

    print_comparison_table(results, preempt=args.preempt)

    if not args.no_charts:
        plot_comparison(results, output_dir=args.output, preempt=args.preempt)


if __name__ == "__main__":
    main()
