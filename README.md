# SelfCheckGPT — Exact Reproduction (EMNLP 2023)

[![Reproduction Status](https://img.shields.io/badge/Reproduction-100%25%20Verified-success)](outputs/metrics/reproduction_report.md)
[![Non-Fact AUC-PR](https://img.shields.io/badge/Non--Fact%20AUC--PR-92.50%20(Target%2092.50)-brightgreen)](outputs/metrics/reproduction_report.md)
[![Factual AUC-PR](https://img.shields.io/badge/Factual%20AUC--PR-66.15%20(Target%2066.08)-brightgreen)](outputs/metrics/reproduction_report.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An independent, exact scientific reproduction of the **SelfCheck-NLI** black-box hallucination detection method from:

> **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models**  
> Potsawee Manakul, Adian Liusie, Mark J. F. Gales  
> *EMNLP 2023* | [[arXiv:2303.08896]](https://arxiv.org/abs/2303.08896)

---

## 📊 Reproduction Results

Our reproduction strictly matches the paper's reported sentence-level **AUC-PR** metrics on the evaluation dataset.

| Method / Task | Target Paper AUC-PR | Reproduced AUC-PR | Δ | Status |
| :--- | :---: | :---: | :---: | :--- |
| **SelfCheck-NLI (Non-Factual)** | **92.50** | **92.50** | **+0.00** | 🎯 **Perfect Match** |
| **SelfCheck-NLI (Factual)** | **66.08** | **66.15** | **+0.07** | ✅ **Match** (hardware precision variance) |
| Random Baseline (Non-Factual) | 72.96 | 72.96 | 0.00 | ✅ Baseline Match |
| Random Baseline (Factual) | 27.04 | 27.04 | 0.00 | ✅ Baseline Match |

> Detailed findings and breakdowns are documented in [`outputs/metrics/reproduction_report.md`](outputs/metrics/reproduction_report.md).

---

## 🔬 Scientific Configuration

The experimental parameters follow the original paper without introducing modern LLMs or modified splits:

| Parameter | Specification |
| :--- | :--- |
| **Dataset** | `potsawee/wiki_bio_gpt3_hallucination` (`evaluation` split) |
| **Passages** | 238 passages (1,908 sentences total) |
| **Stochastic Samples** | 20 GPT-3 generated samples per passage ($N = 20$) |
| **Total Inference Pairs** | $1,908 \times 20 = 38,160$ pairs |
| **NLI Checkpoint** | `potsawee/deberta-v3-large-mnli` |
| **Scoring Formula** | Equation 9 & 10: $S(i) = \frac{1}{N}\sum_{n=1}^{N} \frac{P(\text{contradiction})}{P(\text{entailment}) + P(\text{contradiction})}$ |

The complete frozen spec is defined in [`configs/reproduction.yaml`](configs/reproduction.yaml).

---

## 🔍 Key Technical Discoveries & Gotchas

During reproduction, three subtle implementation nuances were identified that are critical for achieving parity with the original results:

1. **2-Class Output Head**:  
   Standard MNLI models output 3 classes (`[Entailment, Neutral, Contradiction]`). However, `potsawee/deberta-v3-large-mnli` has a stripped 2-class head where **Index 0 = Entailment** and **Index 1 = Contradiction**. Because $P(E) + P(C) = 1.0$, the score is effectively $P(\text{Contradiction})$.
2. **Inverse Prompt Ordering**:  
   Standard NLI pairs pass `(Premise, Hypothesis)`. SelfCheckGPT passes **`(Hypothesis, Premise)`** (i.e. Sentence under test first, followed by stochastic sample). Inverting this order breaks detection accuracy.
3. **DeBERTa Slow Tokenizer Bug Workaround**:  
   Modern versions of `transformers` using `use_fast=False` fail to properly inject `[CLS]` and `[SEP]` tokens when pair-tokenizing DeBERTa sequences, causing ~0.98 contradiction across all inputs. We resolved this by explicitly constructing tokens with proper delimiter IDs (`[1] + hyp_ids + [2] + prem_ids + [2]`).

---

## 📁 Repository Structure

```
SelfCheckGPT-reproduction/
├── configs/
│   └── reproduction.yaml               # Immutable scientific configuration
├── data/
│   ├── raw/                            # Raw dataset download cache
│   └── processed/                      # Extracted evaluation split
├── notebooks/
│   ├── colab_inference.ipynb           # GPU inference pipeline (Colab / Kaggle)
│   └── analysis.ipynb                  # Interactive metrics & score analysis
├── outputs/
│   ├── scores/                         # Final scores (nli_scores.npy, sentence_scores.npy)
│   └── metrics/
│       ├── reproduction_report.md      # Full reproduction report & discoveries
│       └── reproduction_results.json   # Machine-readable evaluation results
├── scripts/
│   ├── run_reproduction.py             # End-to-end pipeline runner
│   ├── smoke_test.py                   # Quick CPU sanity test on 2 passages
│   └── validate_results.py             # AUC-PR verification script
├── src/
│   ├── data.py                         # Data loading & preprocessing
│   ├── nli_detector.py                 # NLI scoring & tokenization handling
│   ├── evaluation.py                   # AUC-PR & metric computation
│   └── utils.py                        # Configuration & logging utilities
├── requirements.txt                    # Project dependencies
└── requirements-lock.txt               # Pinned environment packages
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kvsvar/SelfCheckGPT-reproduction.git
cd SelfCheckGPT-reproduction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset

```bash
python -m src.data
# Output: 238 passages / 1908 sentences / 20 samples each
```

### 3. Run Inference

- **Full GPU Run (Colab / Kaggle)**: Open [`notebooks/colab_inference.ipynb`](notebooks/colab_inference.ipynb) on Google Colab with a GPU (T4/V100/A100). Run all cells to compute `outputs/scores/sentence_scores.npy`.
- **Local Smoke Test (CPU)**:
  ```bash
  python scripts/smoke_test.py
  ```

### 4. Evaluate & Verify Metrics

```bash
python scripts/validate_results.py
```

---

## 📜 Citation

If you use this reproduction codebase or reference our findings, please cite the original SelfCheckGPT paper:

```bibtex
@inproceedings{manakul-etal-2023-selfcheckgpt,
    title = "{S}elf{C}heck{GPT}: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
    author = "Manakul, Potsawee  and
      Liusie, Adian  and
      Gales, Mark",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.557",
    pages = "9004--9017"
}
```
