"""
src/data.py — Dataset loading and flattening for SelfCheckGPT Experiment 1.

Loads the frozen GPT-3 generations from:
    potsawee/wiki_bio_gpt3_hallucination  (evaluation split)

Key functions:
    load_raw()    → raw HuggingFace Dataset object
    flatten()     → list of sentence-level records (see schema below)
    validate_dataset() → assertions + printed report

Record schema:
    {
        "passage_id":   int,   # 0-indexed passage number
        "sentence_idx": int,   # 0-indexed within passage
        "sentence":     str,   # from gpt3_sentences (NOT re-split)
        "label":        str,   # accurate | minor_inaccurate | major_inaccurate
        "samples":      list[str],  # exactly 20 stochastic passages
    }

IMPORTANT: gpt3_sentences[i] <-> annotation[i] alignment is preserved exactly.
           Do NOT re-split sentences; use the provided gpt3_sentences field.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any

from datasets import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

DATASET_NAME = "potsawee/wiki_bio_gpt3_hallucination"
DATASET_SPLIT = "evaluation"

EXPECTED_PASSAGES = 238
EXPECTED_SENTENCES = 1908
EXPECTED_SAMPLES = 20

VALID_LABELS = {"accurate", "minor_inaccurate", "major_inaccurate"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw():
    """
    Load the evaluation split of the WikiBio/GPT-3 hallucination dataset.

    Returns:
        datasets.Dataset: raw HuggingFace Dataset object (238 rows).
    """
    ds = load_dataset(DATASET_NAME)
    return ds[DATASET_SPLIT]


# ── Flattening ─────────────────────────────────────────────────────────────────

def flatten(raw_dataset) -> list[dict[str, Any]]:
    """
    Flatten the passage-level dataset into sentence-level records.

    Each record ties one sentence to its annotation label and to the full
    list of 20 stochastic samples from the same passage.

    Args:
        raw_dataset: HuggingFace Dataset with one row per passage.

    Returns:
        List of dicts matching the schema in the module docstring.
        Expected length: 1908.
    """
    records: list[dict[str, Any]] = []

    for passage_id, row in enumerate(raw_dataset):
        sentences: list[str] = row["gpt3_sentences"]
        annotations: list[str] = row["annotation"]
        samples: list[str] = row["gpt3_text_samples"]

        # Per-passage alignment assertion — hard fail before expensive NLI
        assert len(samples) == EXPECTED_SAMPLES, (
            f"Passage {passage_id}: expected {EXPECTED_SAMPLES} samples, "
            f"got {len(samples)}"
        )
        assert len(sentences) == len(annotations), (
            f"Passage {passage_id}: {len(sentences)} sentences but "
            f"{len(annotations)} annotations — alignment broken"
        )

        for sentence_idx, (sent, label) in enumerate(zip(sentences, annotations)):
            records.append(
                {
                    "passage_id": passage_id,
                    "sentence_idx": sentence_idx,
                    "sentence": sent,
                    "label": label,
                    "samples": samples,  # shared reference; do not mutate
                }
            )

    return records


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_dataset(records: list[dict[str, Any]], raw_dataset=None) -> dict[str, Any]:
    """
    Run all dataset integrity checks and return a validation summary dict.

    Checks:
        1. Passage count == 238
        2. Total sentence count == 1908
        3. Samples per passage == 20 (already asserted in flatten())
        4. Label set is exactly {accurate, minor_inaccurate, major_inaccurate}
        5. No empty sentences
        6. No empty samples
        7. No None/missing values in sentence, label, samples fields

    Args:
        records: output of flatten()
        raw_dataset: optional raw dataset for additional cross-checks

    Returns:
        dict with validation results (for logging/JSON export)
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. Passage count ──────────────────────────────────────────────────────
    passage_ids = {r["passage_id"] for r in records}
    n_passages = len(passage_ids)
    if n_passages != EXPECTED_PASSAGES:
        errors.append(
            f"PASSAGE COUNT: expected {EXPECTED_PASSAGES}, got {n_passages}"
        )

    # ── 2. Sentence count ────────────────────────────────────────────────────
    n_sentences = len(records)
    if n_sentences != EXPECTED_SENTENCES:
        errors.append(
            f"SENTENCE COUNT: expected {EXPECTED_SENTENCES}, got {n_sentences}"
        )

    # ── 3. Sample count per passage ──────────────────────────────────────────
    bad_sample_counts = [
        (r["passage_id"], r["sentence_idx"], len(r["samples"]))
        for r in records
        if len(r["samples"]) != EXPECTED_SAMPLES
    ]
    if bad_sample_counts:
        errors.append(
            f"SAMPLE COUNT: {len(bad_sample_counts)} records with wrong sample count. "
            f"First: passage={bad_sample_counts[0][0]}, sent={bad_sample_counts[0][1]}, "
            f"count={bad_sample_counts[0][2]}"
        )

    # ── 4. Label integrity ───────────────────────────────────────────────────
    label_counter: Counter = Counter(r["label"] for r in records)
    unexpected_labels = set(label_counter.keys()) - VALID_LABELS
    if unexpected_labels:
        errors.append(f"UNEXPECTED LABELS: {unexpected_labels}")

    # ── 5. Empty sentences ───────────────────────────────────────────────────
    empty_sentences = [
        (r["passage_id"], r["sentence_idx"])
        for r in records
        if not r["sentence"] or not r["sentence"].strip()
    ]
    if empty_sentences:
        warnings.append(f"EMPTY SENTENCES: {len(empty_sentences)} found")

    # ── 6. Empty / None samples ──────────────────────────────────────────────
    missing_sample_content = 0
    for r in records:
        for j, s in enumerate(r["samples"]):
            if not s or not s.strip():
                missing_sample_content += 1
    if missing_sample_content:
        warnings.append(f"EMPTY/NONE SAMPLE STRINGS: {missing_sample_content} found")

    # ── 7. None values in key fields ─────────────────────────────────────────
    none_fields = [
        (r["passage_id"], r["sentence_idx"], field)
        for r in records
        for field in ("sentence", "label", "samples")
        if r[field] is None
    ]
    if none_fields:
        errors.append(f"NONE VALUES: {len(none_fields)} found in key fields")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "passages": n_passages,
        "sentences": n_sentences,
        "samples_per_passage": EXPECTED_SAMPLES,
        "label_counts": dict(label_counter),
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }
    return summary


# ── Main (Gate 3 validation report) ────────────────────────────────────────────

if __name__ == "__main__":
    # Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
    import sys, io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("=" * 62)
    print("SelfCheckGPT Experiment 1 — Dataset Validation (Gate 3)")
    print("=" * 62)

    print(f"\nLoading dataset: {DATASET_NAME} [{DATASET_SPLIT}] ...")
    raw = load_raw()
    print(f"  Raw dataset loaded: {len(raw)} rows")

    print("\nFlattening to sentence-level records ...")
    records = flatten(raw)

    print("\nRunning validation checks ...")
    summary = validate_dataset(records, raw_dataset=raw)

    # -- Print Gate 3 report ---------------------------------------------------
    print()
    print("-" * 62)
    print("VALIDATION REPORT")
    print("-" * 62)
    print(f"Dataset : {summary['dataset']}")
    print(f"Split   : {summary['split']}")
    print()
    print(f"Passages         : {summary['passages']}  (expected {EXPECTED_PASSAGES})")
    print(f"Sentences        : {summary['sentences']}  (expected {EXPECTED_SENTENCES})")
    print(f"Samples/passage  : {summary['samples_per_passage']}  (expected {EXPECTED_SAMPLES})")
    print()
    print("Label distribution:")
    for lbl, cnt in sorted(summary["label_counts"].items()):
        print(f"  {lbl:<22}: {cnt}")
    print()

    if summary["warnings"]:
        print("WARNINGS:")
        for w in summary["warnings"]:
            print(f"  [!]  {w}")
        print()

    if summary["errors"]:
        print("ERRORS (GATE 3 FAILED):")
        for e in summary["errors"]:
            print(f"  [FAIL] {e}")
        print()
        print("-" * 62)
        print("GATE 3: FAILED -- investigate before proceeding to NLI")
        print("-" * 62)
        sys.exit(1)
    else:
        print("Missing values   : 0")
        print("Alignment errors : 0")
        print()
        print("-" * 62)
        print("GATE 3: PASSED [OK]")
        print("-" * 62)

    # -- Hard assertions (belt-and-suspenders after printed report) ------------
    assert len({r["passage_id"] for r in records}) == EXPECTED_PASSAGES, \
        "Passage count assertion failed"
    assert len(records) == EXPECTED_SENTENCES, \
        "Sentence count assertion failed"
    assert all(len(r["samples"]) == EXPECTED_SAMPLES for r in records), \
        "Sample count assertion failed"

    print(f"\nAll assertions passed.")
    print(f"  {len({r['passage_id'] for r in records})} passages")
    print(f"  {len(records)} sentences")
    print(f"  {EXPECTED_SAMPLES} samples per passage")
