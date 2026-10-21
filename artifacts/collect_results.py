"""Export only experiment records from a live worker, without bootstrap secrets."""
import argparse
import base64
import json
import shlex
import subprocess
import zlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sid")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    code = """
import json,base64,zlib
from pathlib import Path
root=Path('/tmp/laya-stage/results')
files={str(p.relative_to(root)):p.read_text() for p in root.rglob('*.jsonl')}
print('LAYA_RESULTS='+base64.b64encode(zlib.compress(json.dumps(files).encode())).decode())
"""
    result = subprocess.run([
        "/home/tiger/.merlin-cli/bin/merlin-cli", "--control-plane", "cn",
        "job-v2", "runs", "execute-script", "--json", json.dumps(dict(
            job_run_id=args.sid, exec="python3 -c " + shlex.quote(code), exec_timeout_seconds=30)),
    ], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    marker = next(x for x in result.stdout.splitlines() if x.startswith("LAYA_RESULTS="))
    files = json.loads(zlib.decompress(base64.b64decode(marker.split("=", 1)[1])))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        dest = args.output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        if dest.name == "rank-0.jsonl":
            rows = [json.loads(x) for x in text.splitlines()]
            significant = [r for r in rows if r["event"] in (
                "generation_case", "training_step", "eval_after", "pipeline_case", "suite_complete")]
            print(name, "events", len(rows), "cases", len([r for r in significant if r["event"] == "pipeline_case"]), "last", rows[-1]["event"] if rows else "empty")
    print("exported_files", len(files), "sid", args.sid)


if __name__ == "__main__":
    main()
