"""Send small, compressed commands to a running experiment worker."""
import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import zlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sid")
    p.add_argument("--file", type=Path)
    p.add_argument("--remote", default="/tmp/laya-stage/code/hardware_experiments.py")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--inspect", action="store_true")
    args = p.parse_args()
    statements = ["from pathlib import Path", "import base64,hashlib,zlib,json,time"]
    if args.file:
        data = args.file.read_bytes()
        blob = base64.b64encode(zlib.compress(data)).decode()
        statements += [
            f"data=zlib.decompress(base64.b64decode({blob!r}))",
            f"assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(data).hexdigest()!r}",
            f"p=Path({args.remote!r})", "p.with_suffix('.staged').write_bytes(data)",
            "p.with_suffix('.staged').replace(p)", "print('staged',p)",
        ]
    if args.resume:
        statements.append("Path('/tmp/laya-stage/resume').touch()")
    if args.inspect:
        statements += [
            "for p in Path('/tmp/laya-stage/results').rglob('rank-0.jsonl'):",
            " print('RESULT',p)",
            " print(p.read_text()[-10000:])",
            "for p in Path('/tmp/laya-stage/assets').rglob('*.partial'):",
            " print('DOWNLOAD',p,p.stat().st_size,round(time.time()-p.stat().st_mtime))",
        ]
    result = subprocess.run([
        "/home/tiger/.merlin-cli/bin/merlin-cli", "--control-plane", "cn",
        "job-v2", "runs", "execute-script", "--json", json.dumps(dict(
            job_run_id=args.sid, exec="python3 -c " + shlex.quote("\n".join(statements)),
            exec_timeout_seconds=30,
        )),
    ], capture_output=True, text=True, timeout=60)
    print(result.stdout[-24000:], result.stderr[-2000:])
    if result.returncode:
        raise RuntimeError(f"Worker action failed: {args.sid}")


if __name__ == "__main__":
    main()
