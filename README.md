# autoresearch-rocm

AMD ROCm fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous LLM training optimization on AMD Instinct GPUs.

Ported from [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda) (NVIDIA fork).

## Target Hardware

| GPU | HBM | bf16 TFLOPS | Compute Units |
|-----|-----|-------------|---------------|
| MI300x | 192 GB HBM3 | 1,307 | 304 |
| MI250X | 128 GB HBM2e | 383 | 220 |

Primary target: **AMD Instinct MI300x** on DigitalOcean GPU droplets.

## Quick Start

```bash
# Clone
git clone https://github.com/elementalcollision/autoresearch-rocm.git
cd autoresearch-rocm

# Install PyTorch with ROCm support
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Install project
uv pip install -e '.[all]'

# Download data (10 shards)
uv run prepare.py --num-shards 10

# Run training (auto-detects AMD GPU)
uv run train.py

# Or force ROCm backend
AUTORESEARCH_BACKEND=rocm uv run train_rocm.py
```

## ROCm-Specific Optimizations

- **torch.compile** with `mode="default"` — AMD Triton backend for kernel fusion
- **SDPA attention** — dispatches to CK-based (Composable Kernel) attention
- **bf16 autocast** — native bfloat16 on MI300x matrix cores
- **Muon optimizer** — compiled step functions with on-device scalar tensors
- **192GB HBM3** — memory is rarely the bottleneck; time budget dominates

### Key Difference from CUDA Fork

The CUDA fork uses `torch.compile(mode="reduce-overhead")` which relies on CUDA Graphs. ROCm uses `mode="default"` instead, as CUDA graph capture has limited support on HIP. The AMD Triton backend still provides kernel fusion.

## Results

*Results will be populated after first MI300x benchmarking run.*

| Dataset | val_bpb | tok/sec | MFU | GPU |
|---------|---------|---------|-----|-----|
| climbmix | — | — | — | MI300x |

## Agent Mode (Autonomous Experiments)

```bash
# Store API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run autonomous experiment loop (headless)
uv run -m tui.headless --tag mar20 --max 100

# Or with TUI dashboard
uv run dashboard.py --agent --tag mar20
```

## DigitalOcean Deployment

Deployment scripts live in the companion [DigitalOceanGPU](https://github.com/elementalcollision/DigitalOceanGPU) repo:

```bash
./gpu_provision_amd.sh gpu-worker        # Provision MI300x droplet
./setup_rocm.sh <droplet-ip>             # Bootstrap environment
./rocm_push.sh <droplet-ip>              # Sync code
./rocm_experiment.sh <droplet-ip> climbmix 100  # Launch experiments
./rocm_monitor.sh <droplet-ip> climbmix  # Monitor progress
./rocm_collect.sh <droplet-ip>           # Pull results
./gpu_destroy.sh gpu-worker              # Tear down
```

## Architecture

```
train.py              # Backend dispatcher (auto-detects ROCm vs CUDA)
train_rocm.py         # ROCm training script (agent modifies hyperparameters)
train_cuda.py         # CUDA training script (kept for cross-platform comparison)
prepare.py            # Data prep, tokenizer, evaluation (read-only)
backends/
  __init__.py         # Hardware detection, FLOPS lookup, tier classification
  muon_rocm.py        # Muon+AdamW optimizer for ROCm (torch.compile)
  muon_cuda.py        # Muon+AdamW optimizer for CUDA
tui/
  headless.py         # Headless experiment runner for remote GPU
  orchestrator.py     # Autonomous experiment loop (LLM-driven)
  llm_backend.py      # Claude API integration
  app.py              # Textual TUI dashboard
```

## Related Projects

- [autoresearch](https://github.com/karpathy/autoresearch) — Original (H100)
- [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda) — NVIDIA fork
- [autoresearch (ARM)](https://github.com/elementalcollision/autoresearch) — Apple Silicon fork
