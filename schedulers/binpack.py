from __future__ import annotations
from typing import Optional

from cluster import AllocationPlan, Cluster
from job import Job
from schedulers.base import BaseScheduler


class BinPackScheduler(BaseScheduler):
    def schedule(
        self, cluster: Cluster, job: Job
    ) -> Optional[AllocationPlan]:
        sorted_nodes = sorted(
            cluster.nodes,
            key=lambda n: (len(n.free_gpus), -int(n.name.split("-")[1])),
        )

        for node in sorted_nodes:
            if node.can_fit(job.gpus_needed, job.memory_needed):
                indices = node.allocate(
                    job.job_id, job.gpus_needed, job.memory_needed
                )
                if indices:
                    return [(node, indices)]

        return cluster.allocate_across_nodes(
            job.job_id, job.gpus_needed, job.memory_needed,
            node_order=sorted_nodes,
        )