# SelfCheckGPT Experiment 1 — Action Log

> Append-only. Read this file first on any restart to know exactly where execution stands.
> Format: one entry per step, written before start (IN-PROGRESS) and updated on completion.

---

## [STEP 1] — Project Skeleton Creation | 2026-09-12T22:55
**Status:** DONE
**Gate:** Pre-Gate
**Summary:** Created full project directory structure, all skeleton/stub files, tracking files, frozen config.
**Key outputs:** project_log.md, project_changes.md, configs/reproduction.yaml, requirements.txt, .gitignore, README.md, src/__init__.py, src/data.py, src/nli_detector.py (stub), src/evaluation.py (stub), src/utils.py (stub), scripts/run_reproduction.py (stub), scripts/validate_results.py (stub), notebooks/colab_inference.ipynb (stub), all output dirs with .gitkeep
**Notes:** None

---

## [STEP 2] — Virtual Environment + Dependency Install | 2026-09-12T23:32
**Status:** DONE
**Gate:** Gate 1
**Summary:** Created .venv, installed all dependencies, ran pip freeze, confirmed versions.
**Key outputs:** .venv/, requirements-lock.txt
**Gate 1 result:**
```
Python          : 3.13.9
torch           : 2.14.0+cpu
transformers    : 5.17.0
datasets        : 5.0.1
scikit-learn    : 1.9.1
numpy           : 2.5.3
pandas          : 3.0.5
CUDA available  : False
Device          : CPU only (inference will run on Colab/Kaggle GPU)
```
**Notes:** torch is CPU-only locally. Full NLI inference will run on Colab/Kaggle T4/P100.

---

## [STEP 3] — Dataset Validation | 2026-09-12T23:54
**Status:** DONE
**Gate:** Gate 3
**Summary:** Loaded potsawee/wiki_bio_gpt3_hallucination (evaluation split), flattened to sentence records, ran all integrity checks.
**Key outputs:** None (validation only; no files written)
**Gate 3 result:**
```
Dataset  : potsawee/wiki_bio_gpt3_hallucination
Split    : evaluation
Passages         : 238  (expected 238)  [OK]
Sentences        : 1908  (expected 1908) [OK]
Samples/passage  : 20   (expected 20)   [OK]
Label distribution:
  accurate         : 516
  major_inaccurate : 761
  minor_inaccurate : 631
Missing values   : 0
Alignment errors : 0
GATE 3: PASSED [OK]
```
**Notes:**
- HF Hub unauthenticated warning is harmless (public dataset, no token needed).
- nonfact_labels.sum() = 761 + 631 = 1392 (matches brief spec)
- factual_labels.sum() = 516 (matches brief spec)
- Unicode fix applied to src/data.py for Windows cp1252 console compatibility.

---

## CURRENT STATUS
**Gates passed:** Gate 1 [OK], Gate 3 [OK], Gate 5 [IN-PROGRESS]
**Next gate:** Gate 5 — NLI smoke test
**Awaiting:** Smoke test results.

---

## [STEP 4] — NLI Scorer Implementation + Gate 5 Smoke Test | 2026-09-12T23:57
**Status:** DONE (Kaggle handoff)
**Gate:** Gate 5 (to be run on Kaggle)
**Summary:** Implemented src/nli_detector.py (Eq. 9-10) and notebooks/colab_inference.ipynb (full self-contained Kaggle notebook). Local CPU too slow for ~1.4GB DeBERTa model -- moved inference to Kaggle GPU.
**Key outputs:** src/nli_detector.py, scripts/smoke_test.py, notebooks/colab_inference.ipynb (fully implemented)
**Discrepancies found during this step:**
- Transformers 5.x: potsawee/deberta-v3-large-mnli tokenizer needs sentencepiece + protobuf + use_fast=False
- Fixed: pip install sentencepiece tiktoken protobuf; use_fast=False in AutoTokenizer.from_pretrained()
- [x] (Gate 5) Colab/Kaggle NLI Smoke Test | `notebooks/colab_inference.ipynb` | PASS (Fixed `transformers` tokenizer bug with manual token type building).
- [x] (Gate 6) Full Inference on Kaggle/Colab | Generated `nli_scores.npy` and `sentence_scores.npy`
- [x] (Gate 7/8) Local Metrics Evaluation & Cross-check | Run `scripts/run_reproduction.py` locally and verify AUC-PR against targets.

---

## CURRENT STATUS
**Gates passed:** Gate 1 [OK], Gate 3 [OK], Gate 4 [OK], Gate 5 [OK], Gate 6 [OK], Gate 7/8 [OK]
**Gates pending on Kaggle:** NONE
**Awaiting:** Reviewing final results.
**Next local action:** Finalizing Experiment 1.

