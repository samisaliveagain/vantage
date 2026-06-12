"""Helpers to persist benchmark metrics as Markdown + CSV (committed artifacts).

Keeping machine- and human-readable metrics in the repo makes the project's
claims auditable: every number in the README traces back to a generated file.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return os.path.abspath(path)


def write_markdown_table(path, title, headers, rows, notes=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# {title}", "",
             f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by the eval script — do not edit by hand._", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    if notes:
        lines += ["", "## Notes", "", notes]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.abspath(path)
