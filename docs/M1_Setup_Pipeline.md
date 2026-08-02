# M1 — Environment & Pipeline Setup ✅ COMPLETE

> Ye milestone mukammal ho chuka hai (July 2026). Ye file record ke tor par hai — thesis writing (M6) mein "Implementation Setup" section isi se banega.

## Goal (kya tha)

Local → GitHub → Kaggle pipeline end-to-end verified: code local likha jaye, GitHub par push ho, Kaggle clone kare, install ho, tests pass hon, GPU mile.

## Kya Complete Hua (Definition of Done — sab ✅)

- [x] Local repo: venv, git, folder skeleton (CLAUDE.md section 5 ke mutabiq)
- [x] FederatedScope-LLM `llm` branch as git submodule (`third_party/FederatedScope`)
- [x] `requirements.txt` (local CPU) + `requirements-kaggle.txt` (GPU)
- [x] GA fitness core implemented: `src/ga/fitness.py` — `f(K) = min(A_K, B/(R·K·S))`
- [x] Unit tests passing (locally + Kaggle par)
- [x] `00_kaggle_bootstrap.ipynb`: clone → install → pytest → GPU check
- [x] Kaggle GPU verified: **2× Tesla T4 (15 GB each), CUDA True**
- [x] Results-push branch flow tested (`kaggle-results-run1`)

## Raste Mein Jo Masail Aaye (thesis ke "challenges" section ke liye notes)

1. **Kaggle phone verification** — GPU + Internet toggle dono locked the jab tak phone verify nahi hua.
2. **sentence-transformers conflict warning** — Kaggle ka preinstalled package purane `transformers==4.36.0` pin se lara. Hal: ignore (hum use nahi karte) ya `pip uninstall -y sentence-transformers`.
3. **FederatedScope install failure** — `setup.py` ke 2022-era pins (`numpy<1.23`, `protobuf==3.19.4`) naye Python par source-build fail karte the. Hal: `pip install -e third_party/FederatedScope --no-deps`.
4. **Editable install kernel ko nazar nahi aaya** — running kernel `.pth` nahi parhta. Hal: `sys.path.insert(0, ...)` bootstrap cell mein permanent add kiya.
5. **Empty results/ folder** — git khaali folders track nahi karta; "nothing to commit" normal tha.

## Lesson (aage ke milestones ke liye)

FederatedScope naya environment se larta hai → isliye **Rasta B** (apna FL loop, FS sirf reference) recommended. Faisla M2 se pehle sir se confirm karein.

## Bootstrap Notebook Ka Final Working Order (reference)

```python
# 1. Clone (submodules ke saath)
!git clone --recurse-submodules https://github.com/USERNAME/Lorac-GA.git
%cd /kaggle/working/Lorac-GA

# 2. Deps (conflict warning ignore hogi — normal)
!pip uninstall -y sentence-transformers -q
!pip install -r requirements-kaggle.txt -q

# 3. FederatedScope (no-deps — purane pins bypass)
!pip install -e third_party/FederatedScope --no-deps -q
!pip install fvcore iopath pympler tensorboardX -q

# 4. Kernel import fix
import sys, os
sys.path.insert(0, "/kaggle/working/Lorac-GA/third_party/FederatedScope")
import federatedscope

# 5. Tests + GPU
!python -m pytest tests/ -v
!nvidia-smi
import torch; print("CUDA:", torch.cuda.is_available())
```
