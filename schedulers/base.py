from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from cluster import AllocationPlan, Cluster


class BaseScheduler(ABC):
    @abstractmethod
    def schedule(
        self, cluster: Cluster, job: "Job"
    ) -> Optional[AllocationPlan]:
        ...
