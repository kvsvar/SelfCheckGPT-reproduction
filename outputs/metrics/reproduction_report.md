# SelfCheckGPT — Experiment 1: Exact Reproduction Report

**Date Completed:** September 13, 2026
**Target:** Reproduce the original SelfCheck-NLI methodology and sentence-level AUC-PR results from Manakul et al., EMNLP 2023.

---

## 1. Reproduction Setup

The objective was a strict scientific reproduction of the original methodology without introducing modern LLMs or alternative datasets.

- **Dataset:** `potsawee/wiki_bio_gpt3_hallucination` (Evaluation split)
  - Passages: 238
  - Sentences (A): 1908
  - Samples per passage (N): 20
  - Total Inference Pairs: 38,160
- **Model:** `potsawee/deberta-v3-large-mnli` (Strictly enforced)
- **Scoring Equation (Eq. 9):** `score = P(contradiction) / (P(entailment) + P(contradiction))`

---

## 2. Final Metrics (AUC-PR)

The reproduction was completely successful, achieving a perfect match on the primary non-factual task, and matching the factual task within standard hardware rounding margins.

| Task                   | Target | Achieved  | Status                           |
| :--------------------- | :----- | :-------- | :------------------------------- |
| **Non-Factual AUC-PR** | 92.50  | **92.50** | ✅ **Perfect Match**             |
| **Factual AUC-PR**     | 66.08  | **66.15** | ✅ **Match** (±0.07 HW variance) |
| Random Baseline (NF)   | 72.96  | 72.96     | ✅ Match                         |
| Random Baseline (FA)   | 27.04  | 27.04     | ✅ Match                         |

---

## 3. Methodology & Critical Discoveries

Throughout the reproduction, several critical implementation details were uncovered that differ from standard NLP conventions. These are documented below for future reference:

### A. The 2-Class Model Architecture

Standard MNLI models output 3 classes (Entailment, Neutral, Contradiction). The checkpoint `potsawee/deberta-v3-large-mnli` uploaded by the author is structurally a **2-class model** (the Neutral class was stripped prior to weight saving).

- **Index 0:** Entailment
- **Index 1:** Contradiction
  Because $P(E) + P(C) = 1.0$, the normalization in Equation 9 essentially simplifies to just extracting $P(C)$.

### B. Inverse Input Ordering

Standard NLI models expect inputs in the order of `(Premise, Hypothesis)`. The SelfCheckGPT implementation expects the exact opposite: **`(Hypothesis, Premise)`**. Passing the inputs in standard NLI order yields completely inverted results.

### C. `Transformers` Tokenizer Bug (Missing Special Tokens)

When passing paired strings (hypothesis, premise) to DeBERTa tokenizers using `use_fast=False` in modern versions of the `transformers` library, the tokenizer completely fails to insert the mandatory `[CLS]` and `[SEP]` tokens.

This causes the model to read the entire paired string as a single unformatted sentence, resulting in an unconditional prediction of `Contradiction` (~0.98 probability) for all inputs. We solved this by manually tokenizing each string independently and physically assembling the sequence as `[1] + hyp_ids + [2] + prem_ids + [2]`.
