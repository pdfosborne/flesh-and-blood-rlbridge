#!/usr/bin/env python3
"""Analyze advanced parity sweep results and emit a prioritized generator fix backlog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_sweep_dir(sweep_dir: Path) -> dict[str, Any]:
    summary_path = sweep_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json in {sweep_dir}")

    rows = _load_json(summary_path)
    taxonomy_counts: Counter[str] = Counter()
    card_counts: Counter[str] = Counter()
    matchup_by_taxonomy: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        if row.get("status") != "discrepancy":
            continue
        report_path = row.get("report")
        if not report_path:
            continue
        report_file = Path(report_path)
        if not report_file.is_file():
            continue
        report = _load_json(report_file)
        matchup = f"{row.get('deck1')} vs {row.get('deck2')}"
        for disc in report.get("discrepancies") or []:
            taxonomy = str(disc.get("taxonomy") or disc.get("category") or "unknown")
            taxonomy_counts[taxonomy] += 1
            card_id = str(disc.get("card_id") or "")
            if card_id:
                card_counts[card_id] += 1
            if matchup not in matchup_by_taxonomy[taxonomy]:
                matchup_by_taxonomy[taxonomy].append(matchup)

    failure_index_path = sweep_dir / "failure_index.json"
    failure_index = {}
    if failure_index_path.is_file():
        failure_index = _load_json(failure_index_path)

    backlog_lines = [
        "# Parity Fix Backlog",
        "",
        f"Source sweep: `{sweep_dir}`",
        "",
        "## Failure taxonomy (count)",
        "",
    ]
    for taxonomy, count in taxonomy_counts.most_common():
        backlog_lines.append(f"- **{taxonomy}**: {count}")

    backlog_lines.extend(["", "## Top failing cards", ""])
    if card_counts:
        for card_id, count in card_counts.most_common(25):
            backlog_lines.append(f"- `{card_id}`: {count}")
    else:
        backlog_lines.append("- (no card_id tagged in discrepancies)")

    backlog_lines.extend(["", "## Matchups by taxonomy", ""])
    for taxonomy, matchups in sorted(matchup_by_taxonomy.items()):
        backlog_lines.append(f"### {taxonomy}")
        for matchup in matchups[:10]:
            backlog_lines.append(f"- {matchup}")
        if len(matchups) > 10:
            backlog_lines.append(f"- … {len(matchups) - 10} more")
        backlog_lines.append("")

    backlog_text = "\n".join(backlog_lines) + "\n"
    out_md = sweep_dir / "fix_backlog.md"
    out_md.write_text(backlog_text, encoding="utf-8")

    payload = {
        "taxonomy_counts": dict(taxonomy_counts),
        "card_counts": dict(card_counts),
        "matchup_by_taxonomy": dict(matchup_by_taxonomy),
        "failure_index": failure_index,
    }
    (sweep_dir / "fix_backlog.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        default="",
        help="Path to results/parity_sweeps/advanced_<timestamp>",
    )
    args = parser.parse_args(argv)
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else None
    if sweep_dir is None or not sweep_dir.is_dir():
        sweeps_root = REPO_ROOT / "results" / "parity_sweeps"
        candidates = sorted(sweeps_root.glob("advanced_*"), reverse=True)
        if not candidates:
            print("No advanced sweep directories found.", file=sys.stderr)
            return 2
        sweep_dir = candidates[0]
        print(f"Using latest sweep: {sweep_dir}")

    analyze_sweep_dir(sweep_dir)
    print(f"Wrote {sweep_dir / 'fix_backlog.md'}")
    print(f"Wrote {sweep_dir / 'fix_backlog.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
