"""
run_pipeline.py
────────────────
Master pipeline runner. Orchestrates all stages in order:
  1. generate   — LLM instruction generation (Llama 70B via Ollama)
  2. images     — Image generation (T1/T2/T3/T4 generators)
  3. validate   — Regex validation
  4. inference  — LLaVA scoring (ASR labelling)
  5. write      — HuggingFace Parquet dataset

Usage:
  python run_pipeline.py --config configs/config.yaml
  python run_pipeline.py --config configs/config.yaml --stage generate
  python run_pipeline.py --config configs/config.yaml --stage images
  python run_pipeline.py --config configs/config.yaml --stage inference
  python run_pipeline.py --config configs/config.yaml --stage write
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from rich.logging import RichHandler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("pipeline")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_instructions(config: dict) -> list[dict]:
    path = Path(config["output"]["base_dir"]) / "instructions.jsonl"
    if not path.exists():
        log.error(f"instructions.jsonl not found at {path}. Run --stage generate first.")
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_records(records: list[dict], config: dict):
    """Persist annotated records to a working JSONL for resume support."""
    path = Path(config["output"]["base_dir"]) / "records.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def load_records(config: dict) -> list[dict]:
    path = Path(config["output"]["base_dir"]) / "records.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Stage 1: Generate instructions ───────────────────────────────────────────

def stage_generate(config: dict):
    log.info("═══ Stage 1: Generating instructions (Llama 70B) ═══")
    from orchestrator import OllamaOrchestrator
    orch = OllamaOrchestrator(config)
    orch.generate(resume=True)


# ── Stage 2: Generate images ──────────────────────────────────────────────────

def stage_images(config: dict):
    log.info("═══ Stage 2: Generating images (T1/T2/T3/T4) ═══")

    from generators.t1_typographic import TypographicGenerator
    from generators.t2_structural import StructuralGenerator
    from generators.t3_adversarial import AdversarialGenerator
    from generators.t4_metadata import MetadataGenerator
    from validator import validate_jsonl

    images_dir = Path(config["output"]["images_dir"])
    instructions_path = Path(config["output"]["base_dir"]) / "instructions.jsonl"

    # Validate first to get clean records
    records = validate_jsonl(instructions_path, config)

    # Init generators
    t1 = TypographicGenerator(config, images_dir)
    t2 = StructuralGenerator(config, images_dir)
    t3 = AdversarialGenerator(config, images_dir) if config["attack_types"]["t3_adversarial"]["enabled"] else None
    t4 = MetadataGenerator(config, images_dir)

    # Track generated image paths back into records
    existing = load_records(config)
    existing_ids = {r.get("sample_id") for r in existing if r.get("image_path")}

    failed_imgs = 0
    for i, record in enumerate(tqdm(records, desc="Generating images")):
        sample_id = f"T{record['attack_type']}_{record['goal']}_{i:06d}"
        record["sample_id"] = sample_id

        if sample_id in existing_ids:
            continue  # already done, skip

        at = record["attack_type"]
        instr = record["malicious_instruction"]

        try:
            if at == 1:
                path = t1.generate(instr, sample_id)
                record["image_path"] = str(path)
                record["metadata_fields"] = None

            elif at == 2:
                path = t2.generate(instr, sample_id)
                record["image_path"] = str(path)
                record["metadata_fields"] = None

            elif at == 3:
                if t3 is None:
                    log.warning("T3 disabled in config — skipping.")
                    continue
                path = t3.generate(instr, sample_id)
                record["image_path"] = str(path)
                record["metadata_fields"] = None

            elif at == 4:
                path, meta = t4.generate(instr, sample_id)
                record["image_path"] = str(path)
                record["metadata_fields"] = meta

        except Exception as e:
            log.error(f"Image generation failed for {sample_id}: {e}")
            failed_imgs += 1
            record["image_path"] = None

    log.info(f"Image generation complete. {failed_imgs} failures.")
    save_records(records, config)
    return records


# ── Stage 3: Validate ─────────────────────────────────────────────────────────

def stage_validate(config: dict) -> list[dict]:
    log.info("═══ Stage 3: Regex validation ═══")
    from validator import RegexValidator

    records = load_records(config)
    if not records:
        log.warning("No records found — run --stage images first.")
        return []

    validator = RegexValidator(config)
    for record in records:
        validated = validator.validate_record(record)
        record.update(validated)

    save_records(records, config)
    log.info("Validation complete.")
    return records


# ── Stage 4: Inference (ASR scoring) ─────────────────────────────────────────

def stage_inference(config: dict) -> list[dict]:
    log.info("═══ Stage 4: LLaVA inference (ASR scoring) ═══")
    from inference import LLaVAInferenceRunner

    records = load_records(config)
    if not records:
        log.warning("No records found — run earlier stages first.")
        return []

    runner = LLaVAInferenceRunner(config)
    records = runner.run_batch(records)
    save_records(records, config)
    return records


# ── Stage 5: Write dataset ────────────────────────────────────────────────────

def stage_write(config: dict):
    log.info("═══ Stage 5: Writing HuggingFace dataset ═══")
    from dataset_writer import write_dataset

    records = load_records(config)
    if not records:
        log.warning("No records found — run earlier stages first.")
        return

    out_dir = write_dataset(records, config)
    log.info(f"Dataset written to: {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

STAGES = ["generate", "images", "validate", "inference", "write"]


def run_all(config: dict):
    stage_generate(config)
    records = stage_images(config)
    records = stage_validate(config)
    records = stage_inference(config)
    stage_write(config)


def main():
    parser = argparse.ArgumentParser(description="Image Prompt Injection Dataset Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Create output dirs
    for key in ["base_dir", "images_dir", "dataset_dir", "logs_dir"]:
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)

    start = time.time()

    if args.stage == "all":
        run_all(config)
    elif args.stage == "generate":
        stage_generate(config)
    elif args.stage == "images":
        stage_images(config)
    elif args.stage == "validate":
        stage_validate(config)
    elif args.stage == "inference":
        stage_inference(config)
    elif args.stage == "write":
        stage_write(config)

    elapsed = time.time() - start
    log.info(f"Done in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
