"""
src/nli_detector.py — Independent SelfCheck-NLI scorer (Eq. 9-10).

Implements the SelfCheck-NLI detection method from:
    Manakul, P., Liusie, A., & Gales, M. J. F. (EMNLP 2023)
    "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
     for Generative Large Language Models"

Model: potsawee/deberta-v3-large-mnli  (EXACT checkpoint -- do not substitute)

Formula (Eq. 9):
    score(r_i, s_j) = P(contradiction | premise=s_j, hypothesis=r_i)
                      -------------------------------------------------
                      P(entailment | ...) + P(contradiction | ...)
    Neutral is EXCLUDED from the denominator (per paper).

Formula (Eq. 10):
    S(i) = (1/N) * sum_{j=1}^{N} score(r_i, s_j)

Interpretation:
    S(i) near 1  ->  sentence r_i likely hallucinated
    S(i) near 0  ->  sentence r_i likely factual

IMPORTANT: This module must NOT import or use the official selfcheckgpt package.
           The official package is used ONLY in scripts/validate_results.py (Gate 8).
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "potsawee/deberta-v3-large-mnli"  # EXACT checkpoint -- do not change


# ── Scorer class ───────────────────────────────────────────────────────────────

class SelfCheckNLIScorer:
    """
    Independent implementation of SelfCheck-NLI (Eq. 9-10).

    Usage:
        scorer = SelfCheckNLIScorer()
        scores = scorer.score_batch(premises, hypotheses)  # shape (N,)
    """

    def __init__(self, device: str | None = None):
        """
        Load tokenizer and model. Device defaults to CUDA if available, else CPU.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NLI] Loading tokenizer: {MODEL_NAME}")
        # use_fast=False: forces the slow SentencePiece tokenizer path.
        # Transformers 5.x otherwise incorrectly tries TikToken on the .spm
        # binary file, which fails with a parse error. DeBERTa-v3 uses
        # SentencePiece natively; use_fast=False + sentencepiece + protobuf
        # is the correct loading path for this checkpoint.
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

        print(f"[NLI] Loading model: {MODEL_NAME}")
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        # ── Resolve label indices dynamically -- NEVER hard-code ──────────────
        # The label order varies across NLI checkpoints.
        # We resolve entailment/contradiction indices from model.config.id2label.
        id2label_raw = self.model.config.id2label
        print(f"[NLI] id2label from model config: {id2label_raw}")

        id2label = {k: v.lower() for k, v in id2label_raw.items()}
        entail_candidates = [k for k, v in id2label.items() if "entail" in v]
        contra_candidates = [k for k, v in id2label.items() if "contra" in v]

        if not entail_candidates:
            raise ValueError(
                f"Could not find 'entailment' label in model.config.id2label: {id2label_raw}"
            )
        if not contra_candidates:
            raise ValueError(
                f"Could not find 'contradiction' label in model.config.id2label: {id2label_raw}"
            )

        self.entail_idx: int = entail_candidates[0]
        self.contra_idx: int = contra_candidates[0]
        print(
            f"[NLI] Resolved indices -- "
            f"entailment: {self.entail_idx} ('{id2label_raw[self.entail_idx]}'), "
            f"contradiction: {self.contra_idx} ('{id2label_raw[self.contra_idx]}')"
        )
        print(f"[NLI] Device: {self.device}")

    @torch.inference_mode()
    def score_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Compute per-pair SelfCheck-NLI contradiction scores (Eq. 9).

        Args:
            premises:    List of sampled passages (context).
            hypotheses:  List of target sentences to evaluate.
                         Must be the same length as premises.
            max_length:  Tokenizer truncation length.

        Returns:
            np.ndarray of shape (len(premises),) with values in [0, 1].
            Higher value = more likely contradicted = more likely hallucinated.
        """
        if len(premises) != len(hypotheses):
            raise ValueError(
                f"premises and hypotheses must be the same length, "
                f"got {len(premises)} vs {len(hypotheses)}"
            )
        if not premises:
            return np.array([], dtype=np.float32)

        # Tokenize: premise (sampled passage) as first sequence,
        #           hypothesis (target sentence) as second sequence.
        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits          # shape: (B, num_labels)
        probs = torch.softmax(logits, dim=-1)         # shape: (B, num_labels)

        p_entail = probs[:, self.entail_idx]          # shape: (B,)
        p_contra = probs[:, self.contra_idx]          # shape: (B,)

        # Eq. 9: neutral excluded from denominator
        # Add small epsilon to avoid 0/0 (theoretically impossible but be safe)
        score = p_contra / (p_entail + p_contra + 1e-8)

        return score.cpu().numpy().astype(np.float32)

    def score_sentence(
        self,
        sentence: str,
        samples: list[str],
    ) -> tuple[np.ndarray, float]:
        """
        Compute per-sample scores for one sentence against N samples (Eq. 9),
        then aggregate to a single sentence-level score (Eq. 10).

        Args:
            sentence: The target sentence r_i.
            samples:  List of N stochastic sampled passages.

        Returns:
            (per_sample_scores, sentence_score)
            per_sample_scores: np.ndarray shape (N,)
            sentence_score:    float, mean over N (Eq. 10)
        """
        hypotheses = [sentence] * len(samples)
        per_sample = self.score_batch(samples, hypotheses)
        return per_sample, float(per_sample.mean())
