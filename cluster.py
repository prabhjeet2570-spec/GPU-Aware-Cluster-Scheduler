from __future__ import annotations
from dataclasses import dataclass, field
from statistics import stdev
from typing import Dict, List, Optional, Tuple

AllocationPlan = List[Tuple["Node", List[int]]]


@dataclass
class GPU:
    index: int
    total_memory: int
    allocated_memory: int = 0
    job_id: Optional[int] = None

    @property
    def free_memory(self) -> int:
        return self.total_memory - self.allocated_memory

    @property
    def is_free(self) -> bool:
        return self.job_id is None

    def allocate(self, job_id: int, memory: int) -> None:
        self.job_id = job_id
        self.allocated_memory = memory

    def free(self) -> None:
        self.job_id = None
        self.allocated_memory = 0


@dataclass
class Node:
    name: str
    gpus: List[GPU]

    def __post_init__(self) -> None:
        self._gpu_count = len(self.gpus)

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    @property
    def free_gpus(self) -> List[GPU]:
        return [g for g in self.gpus if g.is_free]

    @property
    def allocated_gpus(self) -> List[GPU]:
        return [g for g in self.gpus if not g.is_free]

    @property
    def utilization(self) -> float:
        if self.gpu_count == 0:
            return 0.0
        return len(self.allocated_gpus) / self.gpu_count

    def can_fit(self, gpus_needed: int, memory_needed: int) -> bool:
        free = self.free_gpus
        if len(free) < gpus_needed:
            return False
        return all(g.free_memory >= memory_needed for g in free[:gpus_needed])

    def allocate(self, job_id: int, gpus_needed: int, memory_needed: int) -> List[int]:
        free = self.free_gpus
        count = 0
        indices: List[int] = []
        for gpu in free:
            if count >= gpus_needed:
                break
            if gpu.free_memory >= memory_needed:
                gpu.allocate(job_id, memory_needed)
                indices.append(gpu.index)
                count += 1
        return indices

    def allocate_specific(self, job_id: int, gpu_indices: List[int], memory: int) -> None:
        for idx in gpu_indices:
            gpu = self.gpus[idx]
            if not gpu.is_free:
                raise ValueError(f"GPU {idx} on {self.name} already allocated")
            if gpu.free_memory < memory:
                raise ValueError(f"GPU {idx} on {self.name} insufficient memory")
            gpu.allocate(job_id, memory)

    def free_job(self, job_id: int) -> None:
        for gpu in self.gpus:
            if gpu.job_id == job_id:
                gpu.free()

    def gpu_indices_for_job(self, job_id: int) -> List[int]:
        return [g.index for g in self.gpus if g.job_id == job_id]


@dataclass
class Cluster:
    nodes: List[Node]

    def total_gpus(self) -> int:
        return sum(n.gpu_count for n in self.nodes)

    def total_memory(self) -> int:
        return sum(g.total_memory for n in self.nodes for g in n.gpus)

    def allocated_gpus(self) -> int:
        return sum(len(n.allocated_gpus) for n in self.nodes)

    def utilization(self) -> float:
        if self.total_gpus() == 0:
            return 0.0
        return self.allocated_gpus() / self.total_gpus()

    def fragmentation(self) -> float:
        total_free = sum(len(n.free_gpus) for n in self.nodes)
        if total_free == 0:
            return 0.0
        max_free_on_node = max((len(n.free_gpus) for n in self.nodes), default=0)
        return 1.0 - (max_free_on_node / total_free)

    def node_utilization_std(self) -> float:
        utils = [n.utilization for n in self.nodes]
        if len(utils) < 2:
            return 0.0
        return stdev(utils)

    def find_node_by_name(self, name: str) -> Optional[Node]:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def free_job(self, job_id: int) -> None:
        for node in self.nodes:
            node.free_job(job_id)

    def allocate_across_nodes(
        self,
        job_id: int,
        gpus_needed: int,
        memory_needed: int,
        node_order: Optional[List[Node]] = None,
    ) -> Optional[List[Tuple[Node, List[int]]]]:
        nodes_to_try = node_order if node_order is not None else self.nodes
        remaining = gpus_needed
        allocations: List[Tuple[Node, List[int]]] = []

        for node in nodes_to_try:
            if remaining <= 0:
                break
            can_take = min(len(node.free_gpus), remaining)
            if can_take > 0:
                indices = node.allocate(job_id, can_take, memory_needed)
                if indices:
                    allocations.append((node, indices))
                    remaining -= len(indices)

        if remaining > 0:
            for node, indices in allocations:
                node.free_job(job_id)
            return None

        return allocations

    def allocate_spread(
        self,
        job_id: int,
        gpus_needed: int,
        memory_needed: int,
        node_order: Optional[List[Node]] = None,
    ) -> Optional[AllocationPlan]:
        nodes_to_try = node_order if node_order is not None else self.nodes
        remaining = gpus_needed
        alloc_map: Dict[str, Tuple[Node, List[int]]] = {}

        while remaining > 0:
            made_progress = False
            for node in nodes_to_try:
                if remaining <= 0:
                    break
                free_with_mem = [
                    g for g in node.free_gpus if g.free_memory >= memory_needed
                ]
                if free_with_mem:
                    gpu = free_with_mem[0]
                    gpu.allocate(job_id, memory_needed)
                    if node.name in alloc_map:
                        alloc_map[node.name][1].append(gpu.index)
                    else:
                        alloc_map[node.name] = (node, [gpu.index])
                    remaining -= 1
                    made_progress = True

            if not made_progress:
                for _, (n, _) in alloc_map.items():
                    n.free_job(job_id)
                return None

        return [
            (n, sorted(indices)) for _, (n, indices) in alloc_map.items()
        ]

    @staticmethod
    def create_uniform(
        node_count: int,
        gpus_per_node: int,
        memory_per_gpu: int,
    ) -> Cluster:
        nodes: List[Node] = []
        for i in range(node_count):
            gpus = [
                GPU(index=j, total_memory=memory_per_gpu)
                for j in range(gpus_per_node)
            ]
            nodes.append(Node(name=f"node-{i}", gpus=gpus))
        return Cluster(nodes=nodes)
