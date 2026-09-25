#!/usr/bin/env python3
"""Recover the exact Protocol 1 eye split embedded in the canonical LMDB."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import lmdb


EXPECTED_COUNTS = {"train": 6114, "val": 1343, "test": 1404}


def subject_of(key: str) -> str:
    return "_".join(key.split("_")[:3])


def eye_of(key: str) -> str:
    return "_".join(key.split("_")[:4])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    split_path = args.output_dir / "fold_0.json"
    manifest_path = args.output_dir / "split_manifest.json"
    if split_path.exists() or manifest_path.exists():
        parser.error(f"refusing to overwrite Protocol 1 split: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    database = lmdb.open(
        str(args.data_dir), readonly=True, lock=False, readahead=False, meminit=False
    )
    with database.begin(write=False) as transaction:
        raw_keys = transaction.get(b"__keys__")
        if raw_keys is None:
            raise KeyError("LMDB does not contain __keys__")
        embedded = pickle.loads(raw_keys)
    database.close()

    keys = {split: list(embedded[split]) for split in ("train", "val", "test")}
    counts = {split: len(split_keys) for split, split_keys in keys.items()}
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"Protocol 1 counts changed: {counts} != {EXPECTED_COUNTS}")

    audits: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        key_overlap = set(keys[left]) & set(keys[right])
        eye_overlap = {eye_of(key) for key in keys[left]} & {
            eye_of(key) for key in keys[right]
        }
        subject_overlap = {subject_of(key) for key in keys[left]} & {
            subject_of(key) for key in keys[right]
        }
        if key_overlap:
            raise AssertionError(f"segment-key overlap {left}/{right}: {len(key_overlap)}")
        if eye_overlap:
            raise AssertionError(f"eye overlap {left}/{right}: {len(eye_overlap)}")
        audits[f"{left}_{right}"] = {
            "segment_key_overlap": 0,
            "eye_overlap": 0,
            "permitted_subject_overlap": len(subject_overlap),
        }

    subjects = {
        split: sorted({subject_of(key) for key in split_keys})
        for split, split_keys in keys.items()
    }
    eyes = {
        split: sorted({eye_of(key) for key in split_keys})
        for split, split_keys in keys.items()
    }
    fold = {
        "protocol": "Protocol 1 (Eye-Level Split)",
        "split_unit": "eye",
        "metric_unit": "segment",
        "subjects": subjects,
        "eyes": eyes,
        "keys": keys,
        "stats": {
            split: {
                "subjects": len(subjects[split]),
                "eyes": len(eyes[split]),
                "samples": len(keys[split]),
            }
            for split in ("train", "val", "test")
        },
    }
    with split_path.open("x") as handle:
        json.dump(fold, handle, indent=2)

    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    manifest = {
        "status": "locked",
        "source_lmdb": str(args.data_dir.resolve()),
        "source": "LMDB __keys__ mapping",
        "expected_counts": EXPECTED_COUNTS,
        "observed_counts": counts,
        "split_unit": "eye",
        "metric_unit": "segment",
        "pairwise_audit": audits,
        "fold_file": split_path.name,
        "fold_file_sha256": digest,
    }
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

