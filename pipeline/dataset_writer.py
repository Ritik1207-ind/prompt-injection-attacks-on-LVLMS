"""
dataset_writer.py
──────────────────
Converts the final list of annotated records into a HuggingFace
`datasets` DatasetDict with train/val/test splits and saves as Parquet.

Output schema:
  image_path        str
  attack_type       int   (1/2/3/4)
  attack_type_name  str
  goal              str
  malicious_instr   str
  benign_context    str
  regex_pattern     str
  regex_hit         bool
  metadata_fields   str   (JSON string; null for T1/T2/T3)
  model_response    str
  asr_label         int   (0/1)
  stealthiness      str
"""

import json
import logging
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
log = logging.getLogger("dataset_writer")

ATTACK_TYPE_NAMES = {
    1: "typographic",
    2: "structural",
    3: "adversarial",
    4: "metadata",
}

STEALTHINESS = {
    1: "zero",
    2: "low",
    3: "maximum",
    4: "maximum",
}

DATASET_FEATURES = Features({
    "image_path":       Value("string"),
    "attack_type":      Value("int32"),
    "attack_type_name": Value("string"),
    "goal":             Value("string"),
    "malicious_instr":  Value("string"),
    "benign_context":   Value("string"),
    "regex_pattern":    Value("string"),
    "regex_hit":        Value("bool"),
    "metadata_fields":  Value("string"),
    "model_response":   Value("string"),
    "asr_label":        Value("int32"),
    "stealthiness":     Value("string"),
})


def _normalise(record: dict) -> dict:
    """Map internal record keys to final dataset schema."""
    at = record.get("attack_type", 0)
    meta = record.get("metadata_fields", None)
    return {
        "image_path":       str(record.get("image_path", "")),
        "attack_type":      int(at),
        "attack_type_name": ATTACK_TYPE_NAMES.get(at, "unknown"),
        "goal":             str(record.get("goal", "")),
        "malicious_instr":  str(record.get("malicious_instruction", "")),
        "benign_context":   str(record.get("benign_context", "")),
        "regex_pattern":    str(record.get("regex_pattern", "")),
        "regex_hit":        bool(record.get("regex_hit", False)),
        "metadata_fields":  json.dumps(meta) if meta else "",
        "model_response":   str(record.get("model_response", "")),
        "asr_label":        int(record.get("asr_label", 0)),
        "stealthiness":     STEALTHINESS.get(at, "unknown"),
    }


def write_dataset(records: list[dict], config: dict) -> Path:
    """
    Takes annotated records, normalises them, splits into train/val/test,
    saves as Parquet, and optionally pushes to HuggingFace Hub.
    Returns path to output dataset directory.
    """
    ds_cfg = config["dataset"]
    out_dir = Path(config["output"]["dataset_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalise
    rows = [_normalise(r) for r in records if r.get("image_path")]
    log.info(f"Writing dataset with {len(rows)} rows.")

    # Build HuggingFace Dataset
    dataset = Dataset.from_list(rows, features=DATASET_FEATURES)
    dataset = dataset.shuffle(seed=42)

    # Split
    train_ratio = ds_cfg.get("train_split", 0.8)
    val_ratio   = ds_cfg.get("val_split", 0.1)
    # test = remainder

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ds = dataset.select(range(n_train))
    val_ds   = dataset.select(range(n_train, n_train + n_val))
    test_ds  = dataset.select(range(n_train + n_val, n))

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    # Save locally as Parquet
    dataset_dict.save_to_disk(str(out_dir))

    # Also export individual Parquet files for easy sharing
    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for split_name, split_ds in dataset_dict.items():
        split_ds.to_parquet(str(parquet_dir / f"{split_name}.parquet"))

    log.info(f"Dataset saved to: {out_dir}")
    log.info(f"  train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Print ASR summary table
    _print_asr_table(rows)

    # Push to HuggingFace Hub
    if ds_cfg.get("push_to_hub") and ds_cfg.get("hub_repo"):
        log.info(f"Pushing to HuggingFace Hub: {ds_cfg['hub_repo']}")
        dataset_dict.push_to_hub(ds_cfg["hub_repo"])
        log.info("Push complete.")

    return out_dir


def _print_asr_table(rows: list[dict]):
    """Print ASR breakdown by attack type × goal."""
    from collections import defaultdict
    stats = defaultdict(lambda: {"success": 0, "total": 0})

    for r in rows:
        key = (r["attack_type_name"], r["goal"])
        stats[key]["total"] += 1
        stats[key]["success"] += r["asr_label"]

    goals = ["jailbreak", "exfiltration", "hijacking", "social_engineering"]
    types = ["typographic", "structural", "adversarial", "metadata"]

    header = f"{'Type':<14}" + "".join(f"{g[:10]:>12}" for g in goals) + f"{'TOTAL':>10}"
    log.info("\n── ASR Table (Attack Success Rate) ──")
    log.info(header)
    log.info("-" * len(header))

    for t in types:
        row_str = f"{t:<14}"
        row_total = 0
        row_success = 0
        for g in goals:
            v = stats.get((t, g), {"success": 0, "total": 0})
            s, tot = v["success"], v["total"]
            pct = s / max(tot, 1) * 100
            row_str += f"{pct:>10.1f}%"
            row_total += tot
            row_success += s
        overall = row_success / max(row_total, 1) * 100
        row_str += f"{overall:>9.1f}%"
        log.info(row_str)
