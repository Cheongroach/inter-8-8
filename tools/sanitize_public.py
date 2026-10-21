#!/usr/bin/env python3
"""Remove infrastructure identifiers while preserving experimental values."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".py", ".txt"}


def pseudonym(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{prefix}-{digest}"


def sanitize(text: str) -> str:
    text = re.sub(
        r"g340-[0-9a-f-]+|n\d{3}-\d{3}-\d{3}",
        lambda match: pseudonym("compute-node", match.group(0)),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:2605|fdbd)(?::[0-9a-f]{0,4}){2,7}",
        lambda match: pseudonym("redacted-ipv6", match.group(0)),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"2605(?:-[0-9a-f]{1,4}){7}",
        lambda match: pseudonym("compute-node", match.group(0)),
        text,
        flags=re.IGNORECASE,
    )
    replacements = (
        (r"fe80(?::[0-9a-f]{0,4}){2,7}", "redacted-link-local"),
        (r"\b26\.(?:\d{1,3}\.){2}\d{1,3}\b", "redacted-ipv4"),
        (r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b", "redacted-mac"),
        (r"GPU-[0-9a-f-]+", "GPU-redacted"),
        (r"trial-\d+(?:-trialrun-\d+)?", "trial-redacted"),
        (r"trial(?:_sid| id|为)?[=:： ]*[0-9]{6,}", "trial-redacted"),
        (r"pod(?:_id| id|为)?[=:： ]*[0-9]{6,}", "pod-redacted"),
        (r"(?:create_)?request_id(?:`|\s|=|:)*[0-9A-F]{20,}`?", "request_id=redacted"),
        (r'"trace_id"\s*:\s*"[^"]+"', '"trace_id": "redacted"'),
        (r'"run_pair"\s*:\s*\[[^\]]+\]', '"run_pair": ["redacted-a", "redacted-b"]'),
        (r'"create_request_id"\s*:\s*"[^"]+"', '"create_request_id": "redacted"'),
        (r'"(?:run_sid|trial_sid|pod_id|pod_sid)"\s*:\s*"?[0-9a-f]+"?', '"platform_id": "redacted"'),
        (r"containerd://[0-9a-f]+", "containerd://redacted"),
        (
            r"https://[^\s)）`]*(?:bytedance|larkoffice|byted\.org)[^\s)）`]*",
            "[internal-reference-omitted]",
        ),
        (r"run为[0-9a-f]{12,}", "run-redacted"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def main(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        original = path.read_text(errors="strict")
        cleaned = sanitize(original)
        if cleaned != original:
            path.write_text(cleaned)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
