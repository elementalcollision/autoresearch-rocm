"""
Autoresearch pretraining script — backend dispatcher.
Detects the best available backend and delegates to the appropriate training script.
Usage: uv run train.py
"""

import os
import sys

# Backend dispatch
_BACKEND = os.environ.get("AUTORESEARCH_BACKEND", "auto").lower()
if _BACKEND == "auto":
    from backends import detect_backend
    _BACKEND = detect_backend()
if _BACKEND == "rocm7":
    os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(__file__), "train_rocm7.py")] + sys.argv[1:])
if _BACKEND == "rocm":
    os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(__file__), "train_rocm.py")] + sys.argv[1:])
if _BACKEND == "cuda":
    os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(__file__), "train_cuda.py")] + sys.argv[1:])

print(f"Error: No supported backend found (detected: {_BACKEND})")
print("This is the ROCm fork. Set AUTORESEARCH_BACKEND=rocm or run train_rocm.py directly.")
sys.exit(1)
