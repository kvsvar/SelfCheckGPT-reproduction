"""
scripts/smoke_test.py — Gate 5: NLI scorer sanity check.

Runs the SelfCheckNLIScorer on a small set of hand-picked
(premise, hypothesis) pairs with obvious expected outcomes.

Expected behavior:
  - Clear contradiction -> score near 1.0
  - Clear entailment    -> score near 0.0
  - Neutral/ambiguous   -> score somewhere in between

Usage:
    python scripts/smoke_test.py

This script downloads potsawee/deberta-v3-large-mnli (~900 MB) on first run.
Subsequent runs use the local HuggingFace cache.
"""

from __future__ import annotations

import sys
import io

# Force UTF-8 stdout for Windows cp1252 compatibility
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

# Add project root to path so we can import src/
sys.path.insert(0, ".")
from src.nli_detector import SelfCheckNLIScorer


# ── Smoke test pairs ───────────────────────────────────────────────────────────
# Format: (premise, hypothesis, expected_direction, description)
# expected_direction: "low"  -> expect score near 0 (entailment / factual)
#                    "high" -> expect score near 1 (contradiction / hallucination)
#                    "mid"  -> expect score roughly in [0.2, 0.8] (neutral/uncertain)

SMOKE_PAIRS = [
    # ── Clear contradictions (hypothesis directly contradicts premise) ──────
    (
        "The sky is blue during the day.",
        "The sky is green during the day.",
        "high",
        "Direct color contradiction",
    ),
    (
        "Albert Einstein was born in Germany in 1879.",
        "Albert Einstein was born in France.",
        "high",
        "Birthplace contradiction",
    ),
    (
        "Water boils at 100 degrees Celsius at sea level.",
        "Water boils at 50 degrees Celsius at sea level.",
        "high",
        "Scientific fact contradiction",
    ),
    # ── Clear entailments (hypothesis is supported by premise) ─────────────
    (
        "The sky is blue during the day.",
        "The sky has a blue color during daytime.",
        "low",
        "Paraphrase entailment",
    ),
    (
        "Albert Einstein was born in Germany in 1879.",
        "Einstein was born in 1879.",
        "low",
        "Subset entailment",
    ),
    (
        "Marie Curie won two Nobel Prizes, one in Physics and one in Chemistry.",
        "Marie Curie was awarded a Nobel Prize in Chemistry.",
        "low",
        "Partial fact entailment",
    ),
    # ── Neutral (hypothesis is neither supported nor contradicted) ──────────
    (
        "The sky is blue during the day.",
        "Astronomers study the stars at night.",
        "mid",
        "Unrelated neutral",
    ),
    (
        "Albert Einstein was born in Germany in 1879.",
        "Einstein had two sons.",
        "mid",
        "Related but neutral",
    ),
    # ── Real dataset-like pairs (mimic actual WikiBio/GPT-3 scenario) ───────
    (
        "John Smith was an American politician who served as governor of Ohio "
        "from 1960 to 1964. He was a member of the Democratic Party.",
        "John Smith served as governor of Ohio.",
        "low",
        "Real-world entailment (factual sentence)",
    ),
    (
        "John Smith was an American politician who served as governor of Ohio "
        "from 1960 to 1964. He was a member of the Democratic Party.",
        "John Smith was a Republican senator from Texas.",
        "high",
        "Real-world contradiction (hallucinated sentence)",
    ),
]


def run_smoke_test():
    print("=" * 66)
    print("Gate 5 Smoke Test -- SelfCheckNLIScorer")
    print("=" * 66)
    print()

    scorer = SelfCheckNLIScorer()
    print()

    premises   = [p for p, _, _, _ in SMOKE_PAIRS]
    hypotheses = [h for _, h, _, _ in SMOKE_PAIRS]
    expected   = [e for _, _, e, _ in SMOKE_PAIRS]
    descs      = [d for _, _, _, d in SMOKE_PAIRS]

    scores = scorer.score_batch(premises, hypotheses)

    print()
    print("-" * 66)
    print(f"{'#':<3}  {'Score':>6}  {'Expected':>8}  {'Pass?':>5}  Description")
    print("-" * 66)

    passed = 0
    failed = 0
    results = []

    for i, (score, exp, desc) in enumerate(zip(scores, expected, descs)):
        if exp == "high":
            ok = score >= 0.5
        elif exp == "low":
            ok = score <= 0.5
        else:  # mid
            ok = 0.1 <= score <= 0.9  # very loose; just checking it's not degenerate

        status = "[OK]  " if ok else "[FAIL]"
        print(f"{i+1:<3}  {score:>6.4f}  {exp:>8}  {status}  {desc}")

        results.append({
            "premise": premises[i][:80] + "..." if len(premises[i]) > 80 else premises[i],
            "hypothesis": hypotheses[i],
            "score": float(score),
            "expected": exp,
            "pass": ok,
        })

        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 66)
    print(f"Passed: {passed}/{len(scores)}   Failed: {failed}/{len(scores)}")
    print()

    # Key stats
    contra_scores = [scores[i] for i, e in enumerate(expected) if e == "high"]
    entail_scores = [scores[i] for i, e in enumerate(expected) if e == "low"]
    neutral_scores = [scores[i] for i, e in enumerate(expected) if e == "mid"]

    print("Score summary by expected class:")
    print(f"  Contradictions (expect high) : mean={np.mean(contra_scores):.4f}  min={np.min(contra_scores):.4f}  max={np.max(contra_scores):.4f}")
    print(f"  Entailments    (expect low)  : mean={np.mean(entail_scores):.4f}  min={np.min(entail_scores):.4f}  max={np.max(entail_scores):.4f}")
    print(f"  Neutral        (expect mid)  : mean={np.mean(neutral_scores):.4f}  min={np.min(neutral_scores):.4f}  max={np.max(neutral_scores):.4f}")
    print()

    # Gate 5 pass condition:
    # All contradiction pairs must score >= 0.5
    # All entailment pairs must score <= 0.5
    # Contradictions must have higher mean than entailments
    strict_pass = (
        all(s >= 0.5 for s in contra_scores) and
        all(s <= 0.5 for s in entail_scores) and
        np.mean(contra_scores) > np.mean(entail_scores)
    )

    if strict_pass:
        print("-" * 66)
        print("GATE 5: PASSED [OK]")
        print("  Score direction is correct: contradiction > entailment")
        print("-" * 66)
        return True
    else:
        print("-" * 66)
        print("GATE 5: FAILED -- score direction is WRONG")
        print("  Check premise/hypothesis ordering and label index resolution.")
        print("  Run the 9-step discrepancy checklist before changing anything.")
        print("-" * 66)
        return False


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
