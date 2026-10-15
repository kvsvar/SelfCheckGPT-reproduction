# SelfCheckGPT — Experiment 1: Exact Reproduction

Independent reproduction of the **SelfCheck-NLI** method from:

> Manakul, P., Liusie, A., & Gales, M. J. F. (2023).  
> *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.*  
> EMNLP 2023. [[arXiv:2303.08896]](https://arxiv.org/abs/2303.08896)

---

## Scientific Configuration (Immutable)

| Parameter | Value |
|-----------|-------|
| Dataset | `potsawee/wiki_bio_gpt3_hallucination` (evaluation split) |
| Passages | 238 |
| Sentences | 1908 |
| Samples/passage | 20 (pre-generated, frozen GPT-3 outputs) |
| NLI model | `potsawee/deberta-v3-large-mnli` |
| Formula | Eq. 9–10 from the paper |
| Target NonFact AUC-PR | 92.50 |
| Target Factual AUC-PR | 66.08 |

See [`configs/reproduction.yaml`](configs/reproduction.yaml) for the complete frozen spec.

## Software Configuration (Modern)

See `requirements-lock.txt` for exact pinned versions after install.  
See `outputs/metrics/run_metadata.json` for runtime hardware/version record.

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip freeze > requirements-lock.txt
```

---

## Running the Pipeline

### Step 1 — Validate dataset (local, CPU)
```bash
python -m src.data
```
Expected output: `238 passages / 1908 sentences / 20 samples each`

### Step 2 — NLI Inference (Colab/Kaggle GPU required)
Upload `notebooks/colab_inference.ipynb` to Google Colab or Kaggle.  
Run all cells. Download `nli_scores.npy` and `sentence_scores.npy` into `outputs/scores/`.

### Step 3 — Evaluation (local, CPU)
```bash
python scripts/run_reproduction.py --skip-inference
```

---

## Output Structure

```
outputs/
├── scores/
│   ├── nli_scores.npy          # (1908, 20) — per-sentence/per-sample contradiction scores
│   └── sentence_scores.npy     # (1908,) — final hallucination scores S(i)
├── metrics/
│   ├── reproduction_results.json
│   ├── reproduction_table.csv
│   └── run_metadata.json
└── logs/
    └── reproduction.log
```

---

## Reproduction Results

*To be filled after Gate 7 (evaluation step).*

| Method | Paper NonFact AUC-PR | Our NonFact AUC-PR | Δ | Paper Factual AUC-PR | Our Factual AUC-PR | Δ |
|--------|---------------------:|-------------------:|--:|---------------------:|-------------------:|--:|
| Random | 72.96 | — | — | 27.04 | — | — |
| SelfCheck-NLI | 92.50 | — | — | 66.08 | — | — |

---

## Progress Log

See [`project_log.md`](project_log.md) for a step-by-step action log.  
See [`project_changes.md`](project_changes.md) for a per-file code change history.
