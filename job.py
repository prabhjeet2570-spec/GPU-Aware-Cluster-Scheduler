from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Job:
    job_id: int
    gpus_needed: int
    memory_needed: int
    duration: int
    priority: int = 0
    submitted_at: int = 0
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    preempted: bool = False
    allocations: List[Tuple[str, List[int]]] = field(default_factory=list)
    partial_start: bool = False

    @property
    def is_running(self) -> bool:
        return self.started_at is not None and self.completed_at is None

    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None

    @property
    def wait_time(self) -> int:
        if self.started_at is None:
            return 0
        return self.started_at - self.submitted_at

    @property
    def nodes_used(self) -> List[str]:
        return [node for node, _ in self.allocations]

    @property
    def gpu_indices(self) -> List[int]:
        indices: List[int] = []
        for _, gpus in self.allocations:
            indices.extend(gpus)
        return indices


class JobGenerator:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(
        self,
        count: int,
        max_gpus: int = 8,
        max_memory: int = 80,
        max_duration: int = 100,
        arrival_window: int = 0,
    ) -> List[Job]:
        jobs: List[Job] = []
        for i in range(count):
            gpus = self._rng.randint(1, max_gpus)
            mem = self._rng.randint(4, max_memory)
            dur = self._rng.randint(5, max_duration)
            priority = self._rng.randint(0, 2)
            submit_time = (
                self._rng.randint(0, arrival_window) if arrival_window > 0 else 0
            )
            jobs.append(
                Job(
                    job_id=i,
                    gpus_needed=gpus,
                    memory_needed=mem,
                    duration=dur,
                    priority=priority,
                    submitted_at=submit_time,
                )
            )
        return jobs
