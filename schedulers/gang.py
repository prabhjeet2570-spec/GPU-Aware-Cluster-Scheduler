from __future__ import annotations
from typing import Optional

from cluster import AllocationPlan, Cluster
from job import Job
from schedulers.base import BaseScheduler


class GangScheduler(BaseScheduler):
    def schedule(
        self, cluster: Cluster, job: Job
    ) -> Optional[AllocationPlan]:
        sorted_nodes = sorted(
            cluster.nodes,
            key=lambda n: (len(n.free_gpus), n.name),
        )

        for node in sorted_nodes:
            if len(node.free_gpus) < job.gpus_needed:
                continue

            free_indices = sorted(g.index for g in node.free_gpus)

            for i in range(len(free_indices) - job.gpus_needed + 1):
                segment = free_indices[i : i + job.gpus_needed]

                is_contiguous = (
                    segment[-1] - segment[0] == len(segment) - 1
                )
                if not is_contiguous:
                    continue

                if all(
                    node.gpus[idx].free_memory >= job.memory_needed
                    for idx in segment
                ):
                    for idx in segment:
                        node.gpus[idx].allocate(job.job_id, job.memory_needed)
                    return [(node, segment)]

        return None
