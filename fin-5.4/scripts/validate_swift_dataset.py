#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path


VALID_ROLES = {"system", "user", "assistant", "tool"}


def load_json_line(raw_line: str, line_no: int, path: Path) -> dict:
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_no} must be a JSON object")
    return record


def validate_messages(messages: object, line_no: int, path: Path) -> dict[str, int]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{line_no} messages must be a non-empty list")

    counts = Counter()
    for idx, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            raise ValueError(f"{path}:{line_no} message #{idx} must be an object")
        role = msg.get("role")
        content = msg.get("content")
        if role not in VALID_ROLES:
            raise ValueError(
                f"{path}:{line_no} message #{idx} has invalid role: {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"{path}:{line_no} message #{idx} content must be a non-empty string"
            )
        counts[role] += 1

    if counts["assistant"] == 0:
        raise ValueError(f"{path}:{line_no} must include at least one assistant message")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{path}:{line_no} last message must be assistant")

    return dict(counts)


def validate_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    records = 0
    total_messages = 0
    role_counts = Counter()
    source_counts = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = load_json_line(line, line_no, path)
            messages = record.get("messages")
            msg_counts = validate_messages(messages, line_no, path)

            records += 1
            total_messages += len(messages)
            role_counts.update(msg_counts)

            source = (
                record.get("source")
                or record.get("dataset")
                or record.get("metadata", {}).get("source")
                or "unknown"
            )
            source_counts[str(source)] += 1

    if records == 0:
        raise ValueError(f"{path} has no valid records")

    return {
        "path": str(path),
        "records": records,
        "avg_messages_per_sample": round(total_messages / records, 2),
        "role_counts": dict(role_counts),
        "top_sources": source_counts.most_common(10),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: python scripts/validate_swift_dataset.py <file1.jsonl> [file2.jsonl ...]",
            file=sys.stderr,
        )
        return 1

    summaries = []
    for raw_path in argv[1:]:
        path = Path(raw_path)
        summary = validate_file(path)
        summaries.append(summary)

    for summary in summaries:
        print(f"[OK] {summary['path']}")
        print(f"  records: {summary['records']}")
        print(f"  avg_messages_per_sample: {summary['avg_messages_per_sample']}")
        print(f"  role_counts: {summary['role_counts']}")
        print(f"  top_sources: {summary['top_sources']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
