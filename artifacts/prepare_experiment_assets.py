"""Download immutable model artifacts and a public workload into a run cache."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests


def event(name, **fields):
    print(json.dumps({"event": name, **fields}), flush=True)


def download(entry, destination):
    path = destination / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == entry["sha256"]:
        event("asset_cached", file=entry["path"])
        return
    for attempt in range(3):
        try:
            with requests.get(entry["url"], stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                digest = hashlib.sha256()
                size = 0
                tmp = path.with_suffix(path.suffix + ".partial")
                with tmp.open("wb") as stream:
                    for chunk in response.iter_content(8 << 20):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                assert size == entry["size"], (entry["path"], size, entry["size"])
                assert digest.hexdigest() == entry["sha256"], entry["path"]
                tmp.replace(path)
                event("asset_verified", file=entry["path"], bytes=size)
                return
        except Exception as error:
            event("asset_retry", file=entry["path"], attempt=attempt, error=str(error))
            if attempt == 2:
                raise
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    for model in manifest["models"]:
        dest = args.destination / model["tag"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda entry: download(entry, dest), model["files"]))
        (dest / "source_manifest.json").write_text(json.dumps(model, indent=2))
    # Use actual GSM8K questions and answers; download once for all hardware.
    dest = args.destination / "data"
    dest.mkdir(exist_ok=True)
    for split in ("train", "test"):
        path = dest / f"{split}.jsonl"
        if not path.exists():
            proxies = {"https": manifest["dataset"]["proxy"], "http": manifest["dataset"]["proxy"]}
            response = requests.get(
                manifest["dataset"]["base_url"] + f"/{split}.jsonl",
                proxies=proxies, timeout=(20, 120),
            )
            response.raise_for_status()
            rows = [json.loads(line) for line in response.text.splitlines()]
            assert all("question" in row and "answer" in row for row in rows)
            temp = path.with_suffix(".partial")
            temp.write_text(response.text)
            temp.replace(path)
        event("dataset_ready", split=split, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              examples=len(path.read_text().splitlines()))
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    event("all_assets_ready", destination=str(args.destination))


if __name__ == "__main__":
    main()
