from __future__ import annotations

from cluster import AllocationPlan, Cluster, GPU, Node
from job import Job, JobGenerator
from schedulers import (
    NaiveScheduler,
    BinPackScheduler,
    SpreadScheduler,
    GangScheduler,
)
from simulation import Simulation


def test_cluster_creation():
    cluster = Cluster.create_uniform(node_count=3, gpus_per_node=4, memory_per_gpu=80)
    assert len(cluster.nodes) == 3
    for node in cluster.nodes:
        assert len(node.gpus) == 4
        for gpu in node.gpus:
            assert gpu.total_memory == 80
            assert gpu.is_free


def test_node_gpu_free_list():
    gpus = [GPU(0, 80), GPU(1, 80), GPU(2, 80)]
    gpus[1].allocate(job_id=1, memory=40)
    node = Node(name="test-node", gpus=gpus)
    assert len(node.free_gpus) == 2
    assert node.free_gpus[0].index == 0
    assert node.free_gpus[1].index == 2
    assert len(node.allocated_gpus) == 1


def test_node_allocate_free():
    c = Cluster.create_uniform(1, 4, 80)
    node = c.nodes[0]
    indices = node.allocate(job_id=1, gpus_needed=2, memory_needed=40)
    assert len(indices) == 2
    for idx in indices:
        assert node.gpus[idx].job_id == 1
        assert node.gpus[idx].allocated_memory == 40
    node.free_job(job_id=1)
    for idx in indices:
        assert node.gpus[idx].is_free


def test_job_creation():
    job = Job(job_id=1, gpus_needed=4, memory_needed=40, duration=50, priority=1)
    assert job.job_id == 1
    assert job.gpus_needed == 4
    assert not job.is_running
    assert not job.is_completed
    assert job.wait_time == 0
    assert len(job.allocations) == 0


def test_job_generator_deterministic():
    gen1 = JobGenerator(seed=42)
    gen2 = JobGenerator(seed=42)
    jobs1 = gen1.generate(count=10)
    jobs2 = gen2.generate(count=10)
    for j1, j2 in zip(jobs1, jobs2):
        assert j1.gpus_needed == j2.gpus_needed
        assert j1.memory_needed == j2.memory_needed
        assert j1.duration == j2.duration


def test_job_generator_different_seeds():
    gen1 = JobGenerator(seed=42)
    gen2 = JobGenerator(seed=99)
    jobs1 = gen1.generate(count=10)
    jobs2 = gen2.generate(count=10)
    diffs = sum(
        1 for j1, j2 in zip(jobs1, jobs2) if j1.gpus_needed != j2.gpus_needed
    )
    assert diffs > 0


def test_naive_fills_nodes_in_order():
    c = Cluster.create_uniform(2, 4, 80)
    s = NaiveScheduler()
    for i in range(6):
        s.schedule(c, Job(i, 1, 10, 10))
    # node-0 fills first (gets 4), then node-1 (gets 2)
    assert len(c.nodes[0].allocated_gpus) == 4
    assert len(c.nodes[1].allocated_gpus) == 2


def test_binpack_packs_highest_index_first():
    c = Cluster.create_uniform(3, 4, 80)
    s = BinPackScheduler()
    for i in range(6):
        s.schedule(c, Job(i, 1, 10, 10))
    # binpack uses reverse-name tiebreaker: node-2 fills first
    assert len(c.nodes[2].allocated_gpus) == 4
    assert len(c.nodes[1].allocated_gpus) == 2


def test_spread_round_robin():
    c = Cluster.create_uniform(3, 4, 80)
    s = SpreadScheduler()
    for i in range(6):
        s.schedule(c, Job(i, 1, 10, 10))
    # spread takes 1 from each node cyclically
    alloc_counts = [len(n.allocated_gpus) for n in c.nodes]
    assert alloc_counts == [2, 2, 2]


def test_naive_cross_node():
    c = Cluster.create_uniform(2, 4, 80)
    s = NaiveScheduler()
    c.nodes[0].allocate(99, 3, 10)
    c.nodes[1].allocate(98, 3, 10)
    r = s.schedule(c, Job(0, 2, 10, 10))
    assert r is not None
    assert len(r) == 2
    assert r[0][0].name == "node-0"
    assert r[1][0].name == "node-1"


def test_naive_no_fit_returns_none():
    c = Cluster.create_uniform(1, 2, 80)
    s = NaiveScheduler()
    c.nodes[0].allocate(0, 2, 40)
    r = s.schedule(c, Job(1, 1, 40, 10))
    assert r is None


def test_bin_pack_prefers_fullest():
    c = Cluster.create_uniform(2, 4, 80)
    s = BinPackScheduler()
    c.nodes[0].allocate(99, 1, 10)
    c.nodes[1].allocate(98, 3, 10)
    # node-1 has 1 free, node-0 has 3 free
    r = s.schedule(c, Job(0, 1, 10, 10))
    assert r is not None
    assert r[0][0].name == "node-1"


def test_spread_prefers_emptiest():
    c = Cluster.create_uniform(2, 4, 80)
    s = SpreadScheduler()
    c.nodes[0].allocate(99, 3, 10)
    c.nodes[1].allocate(98, 1, 10)
    r = s.schedule(c, Job(0, 1, 10, 10))
    assert r is not None
    assert r[0][0].name == "node-1"


def test_gang_single_node():
    c = Cluster.create_uniform(2, 4, 80)
    s = GangScheduler()
    c.nodes[0].allocate(99, 3, 10)
    c.nodes[1].allocate(98, 1, 10)
    r = s.schedule(c, Job(0, 2, 10, 10))
    assert r is not None
    assert len(r) == 1
    assert r[0][0].name == "node-1"


def test_gang_fails_if_no_single_node():
    c = Cluster.create_uniform(2, 4, 80)
    s = GangScheduler()
    c.nodes[0].allocate(99, 3, 10)
    c.nodes[1].allocate(98, 3, 10)
    r = s.schedule(c, Job(0, 2, 10, 10))
    assert r is None


def test_gang_contiguity():
    c = Cluster.create_uniform(1, 4, 80)
    s = GangScheduler()
    node = c.nodes[0]
    node.gpus[0].allocate(99, 10)
    node.gpus[2].allocate(98, 10)
    r = s.schedule(c, Job(0, 2, 10, 10))
    assert r is None


def test_gang_contiguity_success():
    c = Cluster.create_uniform(1, 4, 80)
    s = GangScheduler()
    node = c.nodes[0]
    node.gpus[0].allocate(99, 10)
    node.gpus[3].allocate(98, 10)
    r = s.schedule(c, Job(0, 2, 10, 10))
    assert r is not None
    assert r[0][1] == [1, 2]


def test_fragmentation_metric():
    c = Cluster.create_uniform(1, 4, 80)
    assert c.fragmentation() == 0.0
    c = Cluster.create_uniform(2, 4, 80)
    assert c.fragmentation() == 0.5
    c.nodes[0].allocate(99, 3, 10)
    c.nodes[1].allocate(98, 1, 10)
    assert c.fragmentation() == 0.25


def test_simulation_full_lifecycle():
    c = Cluster.create_uniform(2, 2, 80)
    jobs = [
        Job(0, 1, 10, 5, submitted_at=0),
        Job(1, 1, 10, 5, submitted_at=0),
        Job(2, 1, 10, 5, submitted_at=0),
        Job(3, 1, 10, 5, submitted_at=5),
    ]
    s = NaiveScheduler()
    sim = Simulation(cluster=c, scheduler=s, jobs=jobs)
    r = sim.run()
    assert r.total_jobs == 4
    assert r.completed_jobs == 4
    assert r.total_ticks > 0


def test_simulation_with_preemption():
    c = Cluster.create_uniform(1, 2, 80)
    jobs = [
        Job(0, 2, 10, 100, priority=0, submitted_at=0),
        Job(1, 2, 10, 10, priority=2, submitted_at=10),
    ]
    s = NaiveScheduler()
    sim = Simulation(cluster=c, scheduler=s, jobs=jobs, enable_preemption=True)
    r = sim.run()
    assert r.total_jobs == 2
    assert r.completed_jobs == 2
    assert r.preempted_jobs == 1


def test_gang_vs_naive_different_behavior():
    c = Cluster.create_uniform(2, 4, 80)
    c.nodes[0].allocate(100, 3, 10)
    c.nodes[1].allocate(101, 3, 10)
    j = Job(200, 2, 10, 50)
    naive_result = NaiveScheduler().schedule(c, j)
    c2 = Cluster.create_uniform(2, 4, 80)
    c2.nodes[0].allocate(100, 3, 10)
    c2.nodes[1].allocate(101, 3, 10)
    gang_result = GangScheduler().schedule(c2, j)
    assert naive_result is not None
    assert gang_result is None


def test_node_utilization_std():
    c = Cluster.create_uniform(2, 4, 80)
    assert c.node_utilization_std() == 0.0
    c.nodes[0].allocate(99, 4, 10)
    assert c.node_utilization_std() > 0


def test_spread_round_robin_cross_node():
    c = Cluster.create_uniform(3, 4, 80)
    s = SpreadScheduler()
    r = s.schedule(c, Job(0, 3, 10, 10))
    assert r is not None
    # Should take 1 GPU from each of 3 nodes
    assert len(r) == 3
    nodes_used = [node.name for node, _ in r]
    assert nodes_used == ["node-0", "node-1", "node-2"]


def test_binpack_not_identical_to_naive():
    c1 = Cluster.create_uniform(3, 4, 80)
    c2 = Cluster.create_uniform(3, 4, 80)
    for i in range(4):
        NaiveScheduler().schedule(c1, Job(i, 1, 10, 10))
        BinPackScheduler().schedule(c2, Job(i, 1, 10, 10))
    # First 4 jobs: naive fills node-0, binpack fills node-2
    free_counts_n = [len(n.free_gpus) for n in c1.nodes]
    free_counts_b = [len(n.free_gpus) for n in c2.nodes]
    assert free_counts_n != free_counts_b


if __name__ == "__main__":
    import sys

    tests = [name for name in dir() if name.startswith("test_")]
    failed = 0
    for test_name in tests:
        try:
            globals()[test_name]()
            print(f"  PASS  {test_name}")
        except Exception as e:
            print(f"  FAIL  {test_name}: {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed > 0 else 0)
