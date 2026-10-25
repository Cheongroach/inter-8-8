"""Export a deterministic training prefix and verify the complete test source."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED = {
    "train": ("694aa19971315410e000404523796b76349d3c1d", 29470338),
    "test": ("130b40c3329c993468b8ed8a22c331315e478e0a", 1857427),
}


def verified_source(path, name):
    data = path.read_bytes()
    git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    expected_blob, expected_size = EXPECTED[name]
    return git_blob == expected_blob and len(data) == expected_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-training-prefix-only", action="store_true")
    args = parser.parse_args()
    train_verified = verified_source(args.source / "train.csv", "train")
    if not train_verified and not args.allow_training_prefix_only:
        raise ValueError("Training source Git blob mismatch; complete source required by default")
    if not verified_source(args.source / "test.csv", "test"):
        raise ValueError("Test source Git blob mismatch")
    rows = []
    with (args.source / "train.csv").open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 3 or row[0] not in ("1", "2", "3", "4"):
                raise ValueError("Malformed training source row")
            rows.append(row)
            if len(rows) == 20000:
                break
    if len(rows) != 20000:
        raise ValueError("Training prefix is incomplete")
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite a prepared dataset")
    args.output.mkdir(parents=True)
    with (args.output / "train.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    shutil.copy2(args.source / "test.csv", args.output / "test.csv")
    (args.output / "source.json").write_text(json.dumps({
        "repository": "mhjabreel/CharCnn_Keras",
        "revision": "555590db4219b1243abb1918effd6a7425a2d75f",
        "original_paths": "data/ag_news_csv/{train,test}.csv",
        "expected_source_git_blobs": {name: values[0] for name, values in EXPECTED.items()},
        "training_full_source_verified": train_verified,
        "test_full_source_verified": True,
        "train_scope": "first 20000 source rows",
    }, indent=2))


if __name__ == "__main__":
    main()
