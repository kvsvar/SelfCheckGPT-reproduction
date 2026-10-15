"""
scripts/validate_results.py — Official SelfCheckGPT package cross-check (Gate 8).

STUB — Implemented only after Gate 7 (reproduction table) passes.

This script installs and uses the official `selfcheckgpt` pip package to
independently compute sentence-level scores on the same data, then compares
against our independent implementation outputs.

IMPORTANT: This script is intentionally separate from src/ to make it clear
           that the official package is NOT the primary implementation.
           It is used ONLY here, for cross-validation.

Usage (after Gate 7 passes):
    pip install selfcheckgpt    # in the .venv — separate from primary scorer
    python scripts/validate_results.py
"""

# TODO (Gate 8): Implement:
#
#   Load our scores: np.load("outputs/scores/sentence_scores.npy")
#   Load records from src/data.py
#   Run official SelfCheckNLI.predict() on the same premises/hypotheses
#   Compare:
#       max_abs_diff = np.abs(our_scores - official_scores).max()
#       mean_abs_diff = np.abs(our_scores - official_scores).mean()
#       correlation = np.corrcoef(our_scores, official_scores)[0, 1]
#   Print Gate 8 report
