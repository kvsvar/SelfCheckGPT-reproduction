"""
src/evaluation.py — Label mapping, AUC-PR evaluation, and random baseline.

STUB — Logic will be implemented after Gate 6 (full inference matrix available).

Label mapping (from configs/reproduction.yaml):
    NonFact task : accurate → 0, minor_inaccurate → 1, major_inaccurate → 1
    Factual task : accurate → 1, minor_inaccurate → 0, major_inaccurate → 0

Scoring direction:
    NonFact : use S(i) directly (higher = more hallucinated)
    Factual : use 1 - S(i)     (higher = more factual)
"""

# TODO (Gate 7): Implement:
#
#   build_labels(records)
#       -> nonfact_labels (1908,), factual_labels (1908,)
#
#   factual_scores_from_hallucination(hallucination_scores)
#       -> 1.0 - hallucination_scores
#
#   aggregate(nli_matrix, n_samples=20)
#       -> sentence_scores (1908,)  [Eq. 10: mean over axis=1]
#
#   evaluate(hallucination_scores, nonfact_labels, factual_labels)
#       -> {"nonfact_aucpr": float, "factual_aucpr": float}
#
#   random_baseline(nonfact_labels, factual_labels, seed=42)
#       -> {"nonfact_aucpr": float, "factual_aucpr": float}
