#!/usr/bin/env python3
"""Summarize DSSR choices, backtracking, and class-conditioned acceptance."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "source_label",
    "selected_state",
    "dssr_accepted",
    "final_accepted",
    "alpha",
    "diagnostic_margin_drop",
    "diagnostic_rank_violation",
    "trajectory_stability",
    "selector_ready",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rate(rows: list[dict], field: str) -> float | None:
    return mean([float(bool(row[field])) for row in rows])


def main() -> None:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(REQUIRED_FIELDS - set(row))
            if missing:
                raise ValueError(
                    f"Audit row {line_number} is missing fields: {missing}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("Audit log is empty")

    ready_rows = [row for row in rows if bool(row["selector_ready"])]
    final_rows = [row for row in ready_rows if bool(row["final_accepted"])]
    by_class = defaultdict(list)
    for row in ready_rows:
        by_class[int(row["source_label"])].append(row)

    summary = {
        "input": str(args.input.resolve()),
        "rows": len(rows),
        "selector_ready_rows": len(ready_rows),
        "warmup_fraction": 1.0 - len(ready_rows) / len(rows),
        "dssr_acceptance": rate(ready_rows, "dssr_accepted"),
        "final_acceptance": rate(ready_rows, "final_accepted"),
        "final_acceptance_by_source_class": {
            str(class_index): rate(class_rows, "final_accepted")
            for class_index, class_rows in sorted(by_class.items())
        },
        "selected_state_counts": dict(
            sorted(
                Counter(
                    int(row["selected_state"])
                    for row in ready_rows
                    if bool(row["dssr_accepted"])
                ).items()
            )
        ),
        "alpha_counts": dict(
            sorted(Counter(float(row["alpha"]) for row in final_rows).items())
        ),
        "accepted_diagnostic_margin_drop_mean": mean(
            [float(row["diagnostic_margin_drop"]) for row in final_rows]
        ),
        "accepted_diagnostic_rank_violation_mean": mean(
            [float(row["diagnostic_rank_violation"]) for row in final_rows]
        ),
        "accepted_trajectory_stability_mean": mean(
            [float(row["trajectory_stability"]) for row in final_rows]
        ),
        "accepted_domain_gain_retention_mean": mean(
            [
                float(row["domain_gain_retention"])
                for row in final_rows
                if "domain_gain_retention" in row
            ]
        ),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
