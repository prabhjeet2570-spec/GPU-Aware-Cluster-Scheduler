from __future__ import annotations
from typing import List, Optional, Tuple

from cluster import AllocationPlan, Cluster, Node
from job import Job
from schedulers.base import BaseScheduler


class SpreadScheduler(BaseScheduler):
    def schedule(
        self, cluster: Cluster, job: Job
    ) -> Optional[AllocationPlan]:
        sorted_nodes = sorted(
            cluster.nodes,
            key=lambda n: (-len(n.free_gpus), n.name),
        )
        return cluster.allocate_spread(
            job.job_id, job.gpus_needed, job.memory_needed,
            node_order=sorted_nodes,
        )
