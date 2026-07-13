# GPU-Aware Cluster Scheduler Simulator

## What problem are we solving?

Modern AI and machine learning workloads need GPUs to run. If you have a cluster of machines (called **nodes**) with GPUs inside them, you need a **scheduler** to decide which job goes where. A bad scheduler wastes GPUs, makes jobs wait longer, or leaves some nodes overloaded while others sit idle. This project simulates different scheduling strategies so we can compare them head-to-head without needing real hardware.

## Why are we solving it?

Real GPU clusters are expensive and hard to experiment on. If you want to test a new scheduling idea, you'd have to build or rent a cluster, install software, and run jobs for days. Here, we simulate everything on your laptop in seconds. This lets us answer questions like:

- Does packing jobs onto the fewest nodes (bin-packing) work better than spreading them around?
- Is it worth forcing all GPUs for a job to be on the same machine?
- How much does preemption (kicking out low-priority jobs for high-priority ones) help?
- Which scheduling strategy keeps the cluster most balanced?

## What does each file do?

### `cluster.py` — The cluster model

Defines what a GPU cluster looks like:

- **GPU**: a single GPU with memory (e.g., 80 GB). Can be free or allocated to a job.
- **Node**: a machine with a list of GPUs. Tracks which GPUs are free, can allocate/free GPUs for jobs.
- **Cluster**: a collection of nodes. Tracks overall utilization, fragmentation (how scattered free GPUs are), and provides methods to allocate GPUs across multiple nodes.

### `job.py` — The job model and generator

- **Job**: a workload that needs a certain number of GPUs, memory, and time to run. Has a priority level and tracks when it was submitted, started, and completed.
- **JobGenerator**: creates random jobs with a fixed seed so experiments are reproducible. Each job gets random GPU/memory/duration/priority values.

### `simulation.py` — The simulation engine

Runs the clock tick by tick. On each tick:

1. **Complete** jobs whose duration has expired (frees their GPUs).
2. **Submit** new jobs that have arrived.
3. **Schedule** pending jobs using the chosen scheduler.
4. Track fragmentation, utilization balance, wait times, and other metrics.

Also supports **preemption**: high-priority (level 2) jobs can kick out lower-priority jobs to get GPUs faster.

### `schedulers/` — The scheduling algorithms

#### `base.py`
Abstract parent class. Every scheduler must implement `schedule(cluster, job)` which returns an allocation plan or `None` if the job can't fit.

#### `naive.py` — Naive Scheduler

Goes through nodes in order (node-0, node-1, node-2...) and grabs whatever free GPUs it finds first. If one node doesn't have enough, it spills to the next. Simple, but tends to fill up early nodes first.

#### `binpack.py` — BinPack Scheduler

Tries to pack jobs onto the fullest nodes first (least free GPUs). This means it fills up a few nodes completely before touching others. Good for freeing up entire nodes that can be powered down.

#### `spread.py` — Spread Scheduler

Does the opposite: puts jobs on the emptiest nodes first. When a job needs multiple GPUs, it puts one GPU per node in round-robin fashion. This keeps load balanced across all nodes.

#### `gang.py` — Gang Scheduler

Requires all GPUs for a job to be on the **same node** and **contiguous** (next to each other). This is how some real GPU schedulers work (like SLURM's gang scheduling). It avoids cross-node communication overhead but can reject jobs that would fit if spread across nodes.

### `simulate.py` — Command-line entry point

Lets you run a single simulation with any scheduler and see the results. All settings are controlled via flags:

```bash
python simulate.py --scheduler gang --jobs 500 --nodes 3 --gpus-per-node 4
```

### `benchmark.py` — Multi-run benchmark

Runs all four schedulers across many random seeds, collects statistics, prints a comparison table, and generates a bar chart (`benchmark_comparison.png`). Use this to get reliable, statistically sound results:

```bash
python benchmark.py --jobs 500 --seeds 20 --nodes 5 --gpus-per-node 8
```

### `test_scheduler.py` — Unit tests

24 automated tests that verify models, schedulers, and simulation all work correctly.

## Repository structure

```
.
├── cluster.py              # Cluster, Node, GPU models
├── job.py                  # Job model and JobGenerator
├── simulation.py           # Tick-based simulation engine
├── simulate.py             # CLI for single runs
├── benchmark.py            # Multi-seed benchmark + charts
├── test_scheduler.py       # 24 unit tests
├── schedulers/             # Scheduling algorithms
│   ├── __init__.py         # Exports all schedulers
│   ├── base.py             # Abstract base class
│   ├── naive.py            # First-fit in node order
│   ├── binpack.py          # Pack onto fullest nodes
│   ├── spread.py           # Spread across emptiest nodes
│   └── gang.py             # Same-node + contiguous only
├── benchmark_comparison.png # Generated chart (after benchmark run)
├── .gitignore
└── README.md
```

## Tech stack

| What | Why |
|------|-----|
| **Python 3.10+** | Standard for data/ML tooling |
| **No external deps** (for core) | Pure Python, zero install |
| **matplotlib** (optional) | For benchmark bar charts |
| **dataclasses** | Clean, boilerplate-free models |
| **abc** | Abstract scheduler interface |

The core simulation requires **no external packages**. Just Python 3.10+. For charts, install matplotlib:

```bash
pip install matplotlib
```

## How to reproduce results

### One-shot simulation

```bash
# Default: naive scheduler, 500 jobs, 3 nodes x 4 GPUs
python simulate.py

# Try different schedulers
python simulate.py --scheduler bin-pack
python simulate.py --scheduler spread
python simulate.py --scheduler gang

# Adjust cluster size and workload
python simulate.py --scheduler gang --nodes 5 --gpus-per-node 8 --jobs 1000

# Enable preemption
python simulate.py --scheduler naive --preempt
```

### Full benchmark

```bash
# Compare all schedulers across 20 random seeds (5 nodes x 8 GPUs, 500 jobs)
python benchmark.py

# More jobs + more seeds for tighter results
python benchmark.py --jobs 1000 --seeds 30

# Big cluster (40 GPUs total)
python benchmark.py --nodes 5 --gpus-per-node 8

# Benchmark with preemption enabled
python benchmark.py --preempt

# Benchmark specific schedulers only
python benchmark.py --schedulers naive gang

# Skip chart generation
python benchmark.py --no-charts
```

### Run tests

```bash
python test_scheduler.py
```

All 24 tests should pass with output like `24/24 passed`.
