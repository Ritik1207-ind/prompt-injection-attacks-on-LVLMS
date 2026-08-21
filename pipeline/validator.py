"""
validator.py
─────────────
Validates generated instruction strings against per-goal regex patterns.
Populates the `regex_hit` and `regex_pattern` fields in the dataset record.

Also filters out __FAILED__ placeholder samples from the orchestrator.
"""

import re
import json
import logging
from pathlib import Path

import yaml
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger("validator")


class RegexValidator:
    def __init__(self, config: dict):
        self.patterns = {
            goal: re.compile(pattern)
            for goal, pattern in config["regex_patterns"].items()
        }

    def validate(self, instruction: str, goal: str) -> tuple[bool, str]:
        """
        Returns (hit: bool, pattern_str: str).
        hit = True if instruction matches the goal's regex pattern.
        """
        if goal not in self.patterns:
            return False, ""
        pattern = self.patterns[goal]
        matched = bool(pattern.search(instruction))
        return matched, pattern.pattern

    def validate_record(self, record: dict) -> dict:
        """Add regex_hit and regex_pattern fields to a dataset record."""
        instruction = record.get("malicious_instruction", "")
        goal = record.get("goal", "")
        hit, pattern = self.validate(instruction, goal)
        return {
            **record,
            "regex_hit": hit,
            "regex_pattern": pattern,
        }


def validate_jsonl(instructions_path: Path, config: dict) -> list[dict]:
    """
    Load instructions.jsonl, validate each record, return clean list.
    Filters __FAILED__ placeholders.
    """
    validator = RegexValidator(config)
    records = []
    failed = 0
    total = 0

    with open(instructions_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)

            if obj.get("malicious_instruction") == "__FAILED__":
                failed += 1
                continue

            validated = validator.validate_record(obj)
            records.append(validated)

    hit_count = sum(1 for r in records if r.get("regex_hit"))
    log.info(
        f"Validated {len(records)}/{total} records "
        f"({failed} failed placeholders skipped). "
        f"Regex hit rate: {hit_count}/{len(records)} "
        f"({hit_count/max(len(records),1)*100:.1f}%)"
    )

    # Log per-goal hit rates
    from collections import defaultdict
    goal_counts = defaultdict(lambda: [0, 0])  # [hits, total]
    for r in records:
        g = r.get("goal", "unknown")
        goal_counts[g][1] += 1
        if r.get("regex_hit"):
            goal_counts[g][0] += 1

    for goal, (hits, tot) in goal_counts.items():
        log.info(f"  {goal}: {hits}/{tot} regex hits ({hits/max(tot,1)*100:.1f}%)")

    return records
