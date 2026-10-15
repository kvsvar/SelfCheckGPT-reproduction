# SelfCheckGPT Experiment 1 — Code Change Log

> Append-only. Every file creation or modification is recorded here.
> Read this to understand what code exists and why, without opening every file.

---

## Change 1 — project_log.md | 2026-09-12T22:55
**Action:** CREATE
**File:** `project_log.md`
**Reason:** Step 1 — tracking file, per implementation plan requirement
**What changed:** Initial file created with Step 1 IN-PROGRESS entry
**Dependencies:** None

---

## Change 2 — project_changes.md | 2026-09-12T22:55
**Action:** CREATE
**File:** `project_changes.md`
**Reason:** Step 1 — tracking file, per implementation plan requirement
**What changed:** Initial file created (this file)
**Dependencies:** None

---

## Change 3 — configs/reproduction.yaml | 2026-09-12T22:55
**Action:** CREATE
**File:** `configs/reproduction.yaml`
**Reason:** Step 1 — frozen experiment spec, immutable ground truth
**What changed:** Full YAML spec from Implementation Brief Section 1
**Dependencies:** Referenced by all src/ modules for config values

---

## Change 4 — requirements.txt | 2026-09-12T22:55
**Action:** CREATE
**File:** `requirements.txt`
**Reason:** Step 1 — unpinned human-readable dependency list
**What changed:** torch, transformers, datasets, scikit-learn, numpy, pandas, tqdm
**Dependencies:** Used to create venv; pip freeze produces requirements-lock.txt

---

## Change 5 — .gitignore | 2026-09-12T22:55
**Action:** CREATE
**File:** `.gitignore`
**Reason:** Step 1 — exclude venv, outputs/scores (large .npy), cache, pyc
**What changed:** Standard Python + HuggingFace cache gitignore
**Dependencies:** None

---

## Change 6 — README.md | 2026-09-12T22:55
**Action:** CREATE
**File:** `README.md`
**Reason:** Step 1 — project description and instructions
**What changed:** Initial README with project description, setup instructions, gates overview
**Dependencies:** None

---

## Change 7 — src/__init__.py | 2026-09-12T22:55
**Action:** CREATE
**File:** `src/__init__.py`
**Reason:** Step 1 — makes src/ a Python package for `python -m src.data`
**What changed:** Empty init file
**Dependencies:** Required by src/data.py, src/nli_detector.py, etc.

---

## Change 8 — src/data.py | 2026-09-12T22:55 (STEP 2)
**Action:** CREATE
**File:** `src/data.py`
**Reason:** Task B — dataset loader + flattener + validation block
**What changed:**
- `load_raw()`: loads potsawee/wiki_bio_gpt3_hallucination, evaluation split
- `flatten()`: converts to list of {passage_id, sentence_idx, sentence, label, samples}
- `validate_dataset()`: checks 238/1908/20 + alignment + missing values + label set
- `__main__`: runs validation and prints Gate 3 report
**Dependencies:** Used by scripts/run_reproduction.py, src/evaluation.py

---

## Change 9 — src/nli_detector.py (STUB) | 2026-09-12T22:55
**Action:** CREATE
**File:** `src/nli_detector.py`
**Reason:** Step 1 — skeleton only, logic added at Gate 5
**What changed:** Module docstring + TODO comment
**Dependencies:** Will use transformers, torch

---

## Change 10 — src/evaluation.py (STUB) | 2026-09-12T22:55
**Action:** CREATE
**File:** `src/evaluation.py`
**Reason:** Step 1 — skeleton only, logic added at Gate 7
**What changed:** Module docstring + TODO comment
**Dependencies:** Will use sklearn, numpy, src/data.py

---

## Change 11 — src/utils.py (STUB) | 2026-09-12T22:55
**Action:** CREATE
**File:** `src/utils.py`
**Reason:** Step 1 — skeleton only
**What changed:** Module docstring + TODO comment
**Dependencies:** Will provide logging helpers, JSON saving

---

## Change 12 — scripts/run_reproduction.py (STUB) | 2026-09-12T22:55
**Action:** CREATE
**File:** `scripts/run_reproduction.py`
**Reason:** Step 1 — skeleton only, logic added at Gate 6
**What changed:** Module docstring + TODO comment
**Dependencies:** Will use src/data.py, src/nli_detector.py, src/evaluation.py

---

## Change 13 — scripts/validate_results.py (STUB) | 2026-09-12T22:55
**Action:** CREATE
**File:** `scripts/validate_results.py`
**Reason:** Step 1 — skeleton only, logic added at Gate 8
**What changed:** Module docstring + TODO comment
**Dependencies:** Will use outputs/scores/nli_scores.npy, official selfcheckgpt pkg

---

## Change 14 — notebooks/colab_inference.ipynb (STUB) | 2026-09-12T22:55
**Action:** CREATE
**File:** `notebooks/colab_inference.ipynb`
**Reason:** Step 1 — stub notebook, fully implemented before Gate 6 pass
**What changed:** Minimal valid notebook with TODO cells
**Dependencies:** Will inline/import src/ code for Colab/Kaggle execution

---

## Change 15 — src/data.py | 2026-09-12T23:54
**Action:** MODIFY
**File:** `src/data.py`
**Reason:** Gate 3 run failed with UnicodeEncodeError on Windows cp1252 console (U+2500 box-drawing chars)
**What changed:**
- Added `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` at top of `__main__` block
- Replaced all `─` (U+2500) separators with plain `-` ASCII dashes
- Replaced `✓`/`✗`/`⚠` symbols with `[OK]`/`[FAIL]`/`[!]` ASCII equivalents
**Dependencies:** None (print-only changes; logic unchanged)

---

## Change 16 — requirements-lock.txt + venv | 2026-09-13T00:00
**Action:** MODIFY
**File:** `requirements-lock.txt` (regenerated), `.venv/` (packages added)
**Reason:** Gate 5 smoke test failed — potsawee/deberta-v3-large-mnli tokenizer requires `sentencepiece` (DeBERTa-v3 uses SentencePiece). Transformers 5.x fell back to tiktoken when sentencepiece was absent, which also wasn't installed.
**What changed:**
- `pip install sentencepiece==0.2.2 tiktoken==0.14.0`
- `pip freeze > requirements-lock.txt` (updated with new packages)
**Dependencies:** Required by `src/nli_detector.py` tokenizer loading

---

## Change 17 — scripts/smoke_test.py | 2026-09-12T23:58
**Action:** CREATE
**File:** `scripts/smoke_test.py`
**Reason:** Step 4 / Gate 5 — NLI scorer sanity check before full inference
**What changed:**
- 10 hand-picked (premise, hypothesis) pairs: 4 clear contradictions, 4 clear entailments, 2 neutral
- Includes real WikiBio-like pairs mimicking actual dataset scenario
- Gate pass condition: all contra >= 0.5, all entail <= 0.5, mean(contra) > mean(entail)
- ASCII-safe output (UTF-8 wrapper for Windows cp1252)
**Dependencies:** src/nli_detector.py (SelfCheckNLIScorer)

---

## Change 18 — src/nli_detector.py | 2026-09-13T00:04
**Action:** MODIFY
**File:** `src/nli_detector.py`
**Reason:** Transformers 5.x tries TikToken on .spm binary file when use_fast=True; DeBERTa-v3 needs SentencePiece path
**What changed:**
- Added `use_fast=False` to `AutoTokenizer.from_pretrained()` call
- Added explanatory comment documenting why
**Dependencies:** Requires sentencepiece + protobuf installed (added in Change 16)

---

## Change 19 — notebooks/colab_inference.ipynb | 2026-09-13T00:43
**Action:** MODIFY (stub -> full implementation)
**File:** `notebooks/colab_inference.ipynb`
**Reason:** Local CPU too slow for ~1.4GB DeBERTa model; full inference moved to Kaggle/Colab GPU
**What changed:** Fully implemented 9-cell self-contained notebook:
- Cell 1: GPU check, pip install, BATCH_SIZE auto-select (32 for T4/P100, 8 for CPU)
- Cell 2: Dataset load + flatten + Gate 3 re-verification (all src/data.py logic inlined)
- Cell 3: Tokenizer + model load, dynamic id2label resolution (src/nli_detector.py logic inlined)
- Cell 4: Gate 5 smoke test (6 pairs: 3 contradiction, 3 entailment) -- halts on failure
- Cell 5: Build all 38,160 (premise, hypothesis) pairs
- Cell 6: Batched inference with tqdm progress bar (~20min on T4)
- Cell 7: Reshape to (1908,20), Gate 6 validation
- Cell 8: Save 4 files to /kaggle/working/ (nli_scores.npy, sentence_scores.npy, smoke_test_scores.npy, run_metadata.json)
- Cell 9: Final summary with label counts and next-step instructions
**Dependencies:** All inlined; no src/ upload needed on Kaggle

---

## Change 20 — notebooks/colab_inference.ipynb | 2026-09-13T01:02
**Action:** MODIFY (overwrite)
**File:** `notebooks/colab_inference.ipynb`
**Reason:** Kaggle run crashed in Cell 3 (`IndexError`) because `id2label` returned `LABEL_0` and `LABEL_1` generic names instead of "entailment/contradiction", causing string search to fail. This is discrepancy item 8.
**What changed:** 
- Re-wrote Cell 3 to fall back to an empirical probe (running a known contradiction pair) to resolve indices when labels are generic.
- Added logic for 2-class vs 3-class models to safely handle `P(contra)` vs Eq 9.
**Dependencies:** None

---

## Change 21 — notebooks/colab_inference.ipynb | 2026-09-13T01:10
**Action:** MODIFY (overwrite)
**File:** `notebooks/colab_inference.ipynb`
**Reason:** The empirical probe failed because both contradiction and entailment returned 1. Discovered via official SelfCheckGPT source code that `potsawee/deberta-v3-large-mnli` is a custom 2-class model with the neutral class removed at the head layer, where index 1 = contradiction. Most importantly, it expects inputs to the tokenizer in the order `(hypothesis, premise)` rather than the standard MNLI `(premise, hypothesis)`.
**What changed:** 
- Hard-coded `ENTAIL_IDX = 0` and `CONTRA_IDX = 1` for this specific model, removing the empirical probe.
- Swapped argument order in `score_batch()` (Cell 4) from `tokenizer(premises, hypotheses)` to `tokenizer(hypotheses, premises)` to match the official repo's `batch_encode_plus([sentence, sample])`.
**Dependencies:** None

---

## Change 22 — notebooks/colab_inference.ipynb | 2026-09-13T02:08
**Action:** MODIFY (overwrite)
**File:** `notebooks/colab_inference.ipynb`
**Reason:** The model continued to predict 0.98 for all inputs because the Hugging Face `Transformers` (v4.40+) tokenizer silently failed to insert `[CLS]` and `[SEP]` special tokens when handling string pairs with `use_fast=False` for this specific model architecture.
**What changed:** 
- Rewrote the `score_batch()` function in Cell 4 to completely bypass the buggy tokenization logic.
- Implemented manual sequence building: tokenizing premise/hypothesis separately, and explicitly concatenating `[1] + hyp_ids + [2] + prem_ids + [2]` to guarantee the presence of `[CLS]` and `[SEP]`.
- Implemented robust manual 2D padding.
**Dependencies:** None

---

## Change 23 — scripts/run_reproduction.py | 2026-09-13T03:09
**Action:** MODIFY (overwrite)
**File:** `scripts/run_reproduction.py`
**Reason:** Needed to implement the final Gate 7/8 evaluation logic to compute AUC-PR metrics and verify against targets.
**What changed:** 
- Wrote the full local evaluation pipeline using `scikit-learn` `average_precision_score`.
- Handled label mapping accurately according to the paper (Non-factual uses raw P(contra) as predictor, Factual uses -P(contra) as predictor).
- Executed script locally against the downloaded Kaggle scores.
- Result: **Non-Factual AUC-PR perfectly matches the target (92.50). Factual AUC-PR effectively matches (66.15 achieved vs 66.08 target).**
**Dependencies:** Kaggle outputs in `outputs/scores/`.
