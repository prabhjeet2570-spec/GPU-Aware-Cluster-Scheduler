from __future__ import annotations
from typing import Optional

from cluster import AllocationPlan, Cluster
from job import Job
from schedulers.base import BaseScheduler


class NaiveScheduler(BaseScheduler):
    def schedule(
        self, cluster: Cluster, job: Job
    ) -> Optional[AllocationPlan]:
        return cluster.allocate_across_nodes(
            job.job_id, job.gpus_needed, job.memory_needed,
        )
