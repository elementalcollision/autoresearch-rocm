#!/usr/bin/env python3
"""
Cross-backend comparison: ROCm vs CUDA vs Apple Silicon results.

Reads results from autoresearch-rocm, autoresearch-cuda, and autoresearch (ARM) repos,
compares val_bpb across matching datasets.

Usage:
    uv run compare_backends.py [--cuda-dir /path/to/autoresearch-cuda] [--arm-dir /path/to/autoresearch]
"""

import argparse
import os
import csv
from pathlib import Path

ARM_DEFAULT = os.path.expanduser("~/Claude_Primary/multi-dataset/autoresearch")
CUDA_DEFAULT = os.path.expanduser("~/Claude_Primary/autoresearch-cuda")
ROCM_DIR = Path(__file__).parent


def read_results(results_dir):
    """Read best results per dataset from results/<dataset>/results.tsv files."""
    results = {}
    results_path = Path(results_dir) / "results"
    if not results_path.exists():
        return results

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        tsv_path = dataset_dir / "results.tsv"
        if not tsv_path.exists():
            continue

        dataset = dataset_dir.name
        best_bpb = float("inf")
        best_row = None

        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    bpb = float(row.get("val_bpb", "0"))
                    status = row.get("status", "").strip()
                    if status == "keep" and 0 < bpb < best_bpb:
                        best_bpb = bpb
                        best_row = row
                except (ValueError, TypeError):
                    continue

        if best_row:
            results[dataset] = {
                "val_bpb": best_bpb,
                "memory_gb": float(best_row.get("memory_gb", "0")),
                "description": best_row.get("description", "").strip(),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare ROCm vs CUDA vs Apple Silicon results")
    parser.add_argument("--cuda-dir", default=CUDA_DEFAULT,
                        help="Path to CUDA autoresearch repo")
    parser.add_argument("--arm-dir", default=ARM_DEFAULT,
                        help="Path to Apple Silicon autoresearch repo")
    args = parser.parse_args()

    rocm_results = read_results(ROCM_DIR)
    cuda_results = read_results(args.cuda_dir)
    arm_results = read_results(args.arm_dir)

    all_datasets = sorted(set(rocm_results.keys()) | set(cuda_results.keys()) | set(arm_results.keys()))

    if not all_datasets:
        print("No results found in any repo.")
        return

    # Header
    print(f"{'Dataset':<20} {'ROCm val_bpb':>12} {'CUDA val_bpb':>12} {'ARM val_bpb':>12} {'Best':>8}")
    print("-" * 76)

    for ds in all_datasets:
        rocm = rocm_results.get(ds)
        cuda = cuda_results.get(ds)
        arm = arm_results.get(ds)

        rocm_bpb = f"{rocm['val_bpb']:.6f}" if rocm else "—"
        cuda_bpb = f"{cuda['val_bpb']:.6f}" if cuda else "—"
        arm_bpb = f"{arm['val_bpb']:.6f}" if arm else "—"

        # Determine winner
        candidates = {}
        if rocm:
            candidates["ROCm"] = rocm["val_bpb"]
        if cuda:
            candidates["CUDA"] = cuda["val_bpb"]
        if arm:
            candidates["ARM"] = arm["val_bpb"]

        winner = min(candidates, key=candidates.get) if candidates else "—"

        print(f"{ds:<20} {rocm_bpb:>12} {cuda_bpb:>12} {arm_bpb:>12} {winner:>8}")

    print()

    # Summary
    all_three = [ds for ds in all_datasets if ds in rocm_results and ds in cuda_results]
    if all_three:
        rocm_wins = sum(1 for ds in all_three if rocm_results[ds]["val_bpb"] <= min(
            cuda_results.get(ds, {"val_bpb": float("inf")})["val_bpb"],
            arm_results.get(ds, {"val_bpb": float("inf")})["val_bpb"]))
        print(f"ROCm vs CUDA comparison ({len(all_three)} datasets):")
        print(f"  ROCm wins: {rocm_wins}, CUDA wins: {len(all_three) - rocm_wins}")


if __name__ == "__main__":
    main()
