#!/usr/bin/env python3
"""Create the exact major-style downstream learning-curve subsamples.

The final seed-specific auxiliary and CBraMod components remain fixed; only the number of
Protocol-1 training segments supplied to Stage 1/Stage 2 changes, matching E10's definition.
Validation and held-out test keys remain byte-identical to the locked Protocol-1 split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path


SEEDS = (42, 1234, 3407)  # [fig13b-2027] default kept; override with --seeds
FRACTIONS = (0.10, 0.25, 0.50, 0.75)


def eye(key: str) -> str:
    return "_".join(key.split("_")[:4])


def subject(key: str) -> str:
    return "_".join(key.split("_")[:3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    # [fig13b-2027] only addition: choose which seeds to build (sampling rule unchanged)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error(f"refusing to overwrite learning-curve splits: {args.output_root}")
    source_bytes = args.source_split.read_bytes()
    source = json.loads(source_bytes)
    if source.get("split_unit") != "eye":
        raise AssertionError("expected the locked Protocol-1 eye-level split")

    full_train = source["keys"]["train"]
    manifest = {
        "definition": "major E10-style downstream Stage-1/Stage-2 data fraction",
        "source_split": str(args.source_split.resolve()),
        "source_split_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "full_train_segments": len(full_train),
        "validation_and_test_unchanged": True,
        "components_fixed_per_seed": True,
        "entries": [],
    }
    for seed in args.seeds:
        for fraction in FRACTIONS:
            n = max(1, int(len(full_train) * fraction))
            chosen = random.Random(seed).sample(full_train, n)
            split = copy.deepcopy(source)
            split["keys"]["train"] = chosen
            split["eyes"]["train"] = sorted({eye(key) for key in chosen})
            split["subjects"]["train"] = sorted({subject(key) for key in chosen})
            split["stats"]["train"] = {
                "subjects": len(split["subjects"]["train"]),
                "eyes": len(split["eyes"]["train"]),
                "samples": len(chosen),
            }
            split["learning_curve"] = {
                "fraction": fraction, "seed": seed,
                "sampling": "random.Random(seed).sample(full_train, floor(N*fraction))",
                "scope": "Stage 1 and Stage 2 only; seed-specific encoders fixed",
            }
            out_dir = args.output_root / f"seed{seed}" / f"fraction_{int(fraction * 100):02d}"
            out_dir.mkdir(parents=True)
            out_path = out_dir / "fold_0.json"
            out_path.write_text(json.dumps(split, indent=2) + "\n")
            # Core leakage and comparability guards.
            assert not set(chosen).intersection(split["keys"]["val"])
            assert not set(chosen).intersection(split["keys"]["test"])
            assert split["keys"]["val"] == source["keys"]["val"]
            assert split["keys"]["test"] == source["keys"]["test"]
            manifest["entries"].append({
                "seed": seed, "fraction": fraction, "n_train": n,
                "split": str(out_path.resolve()),
                "train_keys_sha256": hashlib.sha256(
                    "\n".join(chosen).encode()).hexdigest(),
            })
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

