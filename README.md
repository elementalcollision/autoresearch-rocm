# autoresearch-rocm

AMD ROCm fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous LLM-driven experiment optimization on AMD Instinct GPUs.

Ported from [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda) (NVIDIA fork). See the [wiki](https://github.com/elementalcollision/autoresearch-rocm/wiki) for detailed experiment results, hardware comparisons, and ROCm software configuration.

## Results

| Dataset | Baseline val_bpb | Best val_bpb | Improvement | tok/s | MFU | Experiments |
|---------|-------------------|--------------|-------------|-------|-----|-------------|
| **climbmix** | 1.066970 | **1.035937** | -2.91% | 397,867 | 13.1% | 100 |
| **fineweb-edu** | 1.035989 | **1.014783** | -2.05% | 549,816 | 18.3% | 100 |

**Key finding:** The MI300x's throughput (~400K tok/s) enables 911 training steps in a 5-minute window vs ~200 on an RTX 4000 Ada, producing lower absolute val_bpb despite the CUDA fork showing a larger relative improvement.

## Target Hardware

| GPU | Architecture | HBM | bf16 TFLOPS | Compute Units |
|-----|-------------|-----|-------------|---------------|
| **MI300X** | CDNA 3 (gfx942) | 192 GB HBM3 | 1,307 | 304 |
| MI325X | CDNA 3 | 256 GB HBM3e | 2,615 | 304 |
| MI350X | CDNA 4 | TBD | ~2,300 (est.) | TBD |
| MI250X | CDNA 2 | 128 GB HBM2e | 383 | 220 |

Primary target: **AMD Instinct MI300X** on RunPod cloud (1x MI300X, ~$1.99/hr).

## Environment

| Component | Version |
|-----------|---------|
| **ROCm** | 6.3 (HIP 6.3.42134) |
| **PyTorch** | 2.9.1+rocm6.3 |
| **Triton** | 3.5.1 (pytorch-triton-rocm) |
| **Python** | 3.10.14 |
| **OS** | Ubuntu 22.04.4 LTS |

## Quick Start

```bash
# Clone
git clone https://github.com/elementalcollision/autoresearch-rocm.git
cd autoresearch-rocm

# Install PyTorch with ROCm 6.3 support
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3

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
- **SDPA attention** — dispatches to CK-based (Composable Kernel) FlashAttention
- **bf16 autocast** — native bfloat16 on MI300x matrix cores (CDNA 3)
- **Muon optimizer** — compiled step functions with on-device scalar tensors
- **192GB HBM3** — memory is rarely the bottleneck; the 5-minute time budget dominates

### ROCm 6.x vs 7.x

Two separate training scripts — 6.x is frozen during active data collection, 7.x adds HIP graph capture and explicit CK backend selection.

| Feature | ROCm 6.x (`train_rocm.py`) | ROCm 7.x (`train_rocm7.py`) |
|---------|---------------------------|---------------------------|
| torch.compile | `mode="default"` | `mode="reduce-overhead"` (HIP graphs) |
| Flash Attention | Auto-selected by SDPA | CK explicitly selected via `preferred_rocm_fa_library("ck")` |
| Backend env var | `AUTORESEARCH_BACKEND=rocm` | `AUTORESEARCH_BACKEND=rocm7` |
| Status | **Production** (active data collection) | Ready for validation after 6.x suite completes |

```bash
# ROCm 7.x usage
AUTORESEARCH_BACKEND=rocm7 uv run train.py
```

## Agent Mode (Autonomous Experiments)

```bash
# Store API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run autonomous experiment loop (headless)
uv run -m tui.headless --dataset climbmix --tag mar20 --max 100

# Or with TUI dashboard
uv run dashboard.py --agent --tag mar20
```

### Multi-Dataset Suite

```bash
# Run full 7-dataset suite
uv run run_suite.py --tag mar22 --max-experiments 100

# Run a single dataset
uv run run_suite.py --dataset fineweb-edu --max-experiments 80

# Check status across all datasets
uv run run_suite.py --status
```

### Model Isolation

Non-default LLM models get isolated results directories for A/B comparison:

```bash
# Default model (Sonnet 4) — results go to results/<dataset>/
uv run run_suite.py --dataset climbmix --max-experiments 100

# Non-default model — results go to results/<model-slug>/<dataset>/
uv run run_suite.py --dataset climbmix --model claude-sonnet-4-6 --max-experiments 100
# Results: results/sonnet-4-6/climbmix/results.tsv
```

### Deployment Fencing

Each deployment writes a `manifest.json` with GPU fingerprint, ROCm version, and provenance metadata. Every experiment row includes a `gpu_name` field. Results TSVs are git-ignored to prevent cross-GPU data contamination when code is cloned to a different instance.

## Architecture

```
train.py              # Backend dispatcher (auto-detects ROCm vs CUDA)
train_rocm.py         # ROCm 6.x training script (production, frozen during data collection)
train_rocm7.py        # ROCm 7.x training script (reduce-overhead compile, CK FA backend)
train_cuda.py         # CUDA training script (kept for cross-platform comparison)
prepare.py            # Data prep, tokenizer, evaluation (read-only)
backends/
  __init__.py         # Hardware detection, FLOPS lookup, tier classification, ROCm version detection
  muon_rocm.py        # Muon+AdamW optimizer for ROCm 6.x (torch.compile)
  muon_rocm7.py       # Muon+AdamW optimizer for ROCm 7.x
  muon_cuda.py        # Muon+AdamW optimizer for CUDA
tui/
  headless.py         # Headless experiment runner for remote GPU
  orchestrator.py     # Autonomous experiment loop (LLM-driven)
  llm_backend.py      # Claude API integration (dynamic system prompt)
  results.py          # Results management with gpu_name fingerprinting
  resilience.py       # Crash recovery, atomic I/O, heartbeat
  credentials.py      # API key management (keychain + env)
  app.py              # Textual TUI dashboard
run_suite.py          # Multi-dataset suite runner with model isolation
results/
  <dataset>/          # Default model results (git-ignored TSVs)
  <model-slug>/       # Non-default model results (e.g. sonnet-4-6/)
    <dataset>/
```

## Related Projects

- [autoresearch](https://github.com/karpathy/autoresearch) — Original (H100)
- [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda) — NVIDIA fork
- [autoresearch (Apple Silicon)](https://github.com/elementalcollision/autoresearch) — Apple Silicon fork
- [ROCm Documentation](https://rocm.docs.amd.com/) — AMD ROCm platform docs
