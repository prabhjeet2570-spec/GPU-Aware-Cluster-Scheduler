from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from cluster import AllocationPlan, Cluster, Node
from job import Job
from schedulers.base import BaseScheduler


@dataclass
class SimulationResult:
    total_jobs: int = 0
    completed_jobs: int = 0
    preempted_jobs: int = 0
    total_ticks: int = 0
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
    jobs_scheduled_per_tick: float = 0.0
    schedule_attempts: int = 0
    schedule_failures: int = 0


class Simulation:
    def __init__(
        self,
        cluster: Cluster,
        scheduler: BaseScheduler,
        jobs: List[Job],
        max_ticks: int = 10_000,
        enable_preemption: bool = False,
    ):
        self.cluster = cluster
        self.scheduler = scheduler
        self.jobs = sorted(jobs, key=lambda j: j.submitted_at)
        self.max_ticks = max_ticks
        self.enable_preemption = enable_preemption
        self.pending: List[Job] = []
        self.running: List[Job] = []
        self.completed: List[Job] = []
        self.current_tick = 0
        self.job_queue: List[Job] = list(self.jobs)
        self.schedule_attempts = 0
        self.schedule_failures = 0
        self.peak_fragmentation = 0.0
        self.fragmentation_samples: List[float] = []
        self.node_std_samples: List[float] = []
        self.peak_node_std = 0.0

    def _complete_jobs(self) -> bool:
        freed = False
        for job in self.running[:]:
            if self.current_tick >= job.started_at + job.duration:
                self.cluster.free_job(job.job_id)
                job.completed_at = self.current_tick
                self.running.remove(job)
                self.completed.append(job)
                freed = True
        return freed

    def _submit_jobs(self) -> bool:
        submitted = False
        while self.job_queue and self.job_queue[0].submitted_at <= self.current_tick:
            self.pending.append(self.job_queue.pop(0))
            submitted = True
        return submitted

    def _try_schedule(self, job: Job) -> bool:
        self.schedule_attempts += 1
        result = self.scheduler.schedule(self.cluster, job)
        if result is not None:
            job.allocations = [
                (node.name, indices) for node, indices in result
            ]
            job.started_at = self.current_tick
            if len(job.allocations) > 1:
                job.partial_start = True
            self.running.append(job)
            return True
        self.schedule_failures += 1
        return False

    def _preempt_for(self, job: Job) -> bool:
        if not self.enable_preemption or job.priority < 2:
            return False

        preemptable = [
            j
            for j in self.running
            if j.priority < job.priority
        ]
        preemptable.sort(key=lambda j: (j.priority, j.started_at or 0))

        freed_gpus = 0
        to_preempt: List[Job] = []
        for j in preemptable:
            freed_gpus += j.gpus_needed
            to_preempt.append(j)
            if freed_gpus >= job.gpus_needed:
                break

        if freed_gpus < job.gpus_needed:
            return False

        for j in to_preempt:
            self.cluster.free_job(j.job_id)
            j.preempted = True
            j.started_at = None
            j.allocations = []
            self.running.remove(j)
            self.pending.append(j)

        return self._try_schedule(job)

    def _schedule_pending(self) -> None:
        still_pending: List[Job] = []
        for job in self.pending:
            if not self._try_schedule(job):
                if not self._preempt_for(job):
                    still_pending.append(job)
        self.pending = still_pending

    def step(self) -> None:
        resources_freed = self._complete_jobs()
        new_jobs_arrived = self._submit_jobs()

        if new_jobs_arrived or resources_freed or self.current_tick == 0:
            self._schedule_pending()

        frag = self.cluster.fragmentation()
        self.fragmentation_samples.append(frag)
        if frag > self.peak_fragmentation:
            self.peak_fragmentation = frag

        nstd = self.cluster.node_utilization_std()
        self.node_std_samples.append(nstd)
        if nstd > self.peak_node_std:
            self.peak_node_std = nstd

        self.current_tick += 1

    def run(self) -> SimulationResult:
        while self.current_tick < self.max_ticks:
            self.step()
            if not self.job_queue and not self.pending and not self.running:
                break

        result = SimulationResult()
        all_jobs = self.completed + self.running + self.pending
        result.total_jobs = len(all_jobs)
        result.completed_jobs = len(self.completed)
        result.preempted_jobs = sum(1 for j in all_jobs if j.preempted)
        result.total_ticks = self.current_tick

        if self.completed:
            wait_times = [j.wait_time for j in self.completed]
            result.avg_wait_time = sum(wait_times) / len(wait_times)
            result.max_wait_time = max(wait_times)
            completion_times = [
                j.completed_at - j.submitted_at for j in self.completed
            ]
            result.avg_completion_time = sum(completion_times) / len(completion_times)

        result.final_fragmentation = self.cluster.fragmentation()
        result.peak_fragmentation = self.peak_fragmentation
        result.avg_fragmentation = (
            sum(self.fragmentation_samples) / len(self.fragmentation_samples)
            if self.fragmentation_samples
            else 0.0
        )

        result.final_node_std = self.cluster.node_utilization_std()
        result.peak_node_std = self.peak_node_std
        result.avg_node_std = (
            sum(self.node_std_samples) / len(self.node_std_samples)
            if self.node_std_samples
            else 0.0
        )

        result.cross_node_jobs = sum(
            1 for j in self.completed if len(j.allocations) > 1
        )

        result.jobs_scheduled_per_tick = (
            result.completed_jobs / result.total_ticks if result.total_ticks > 0 else 0
        )
        result.schedule_attempts = self.schedule_attempts
        result.schedule_failures = self.schedule_failures

        return result
