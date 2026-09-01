# M2 — Federated LoRA Baseline (FedAvg, Fixed K) — Session 2

> **Goal ek line mein:** DistilBERT + LoRA ko GLUE SST-2 par, 100-client non-IID split ke saath, FIXED K=10 par FedAvg se train karke pehli asal accuracy hasil karna. **GA is milestone mein nahi hai.**

## Status: ✅ COMPLETE (Kaggle run 2026-09-01, commit `9d48854`)

## Definition of Done

- [x] SST-2 non-IID Dirichlet split (α=0.3, 100 clients) — shards **min=1, max=4733, mean=673.5**; bohot se clients tuqreeban single-class (`[193,0]`, `[2423,8]`, `[0,1148]`) → `split_summary.png`
- [x] LoRA wrapper: trainable **739,586 / 67,694,596 = 1.093%**, **S = 2.9583 MB** (DistilBERT, r=8)
- [x] Apna FedAvg loop: R=10, K=10, har round accuracy + comm logged, har round checkpoint + resume
- [x] Accuracy: **85.21% final** (best 85.44% @ round 9) — 80–88% target band ke andar
- [x] Results `results/m2_baseline/` mein (results.json + history.json + 2 PNG), GitHub par pushed
- [x] Unit tests: split + LoRA wrapper + aggregation + round + resume — **115 passing**

## M2 Final Numbers (thesis ke liye)

| Metric | Value |
|---|---|
| Model | DistilBERT-base-uncased + LoRA (r=8, α=16, dropout=0.05, targets `q_lin`/`v_lin`) |
| Task | GLUE SST-2 (eval = official `validation`, 872 samples; `test` ke labels −1 hain) |
| Clients | 100, Dirichlet α=0.3, K=10 per round, R=10 rounds |
| Trainable params | 739,586 / 67,694,596 = **1.093%** |
| **S (per-client payload)** | **2.9583 MB** |
| Per-round comm | 2·K·S = **59.17 MB** |
| **Total comm** | **591.67 MB** |
| **Final accuracy** | **85.21%** (best 85.44% @ round 9) |
| Runtime | **9 min 43 s** on 1× Tesla T4 |
| Seed | 42 (single seed — M5 mein multi-seed hoga) |

Per-round accuracy: 0.634, 0.688, 0.611, **0.841**, 0.811, 0.839, 0.849, 0.844, **0.854**, 0.852

### Do observations (M3/M4 ke liye ahem)

1. **Accuracy round 4 par saturate ho jati hai.** Rounds 5–10 ne ~1% diya lekin comm 236 MB se 592 MB tak barh gaya. Cost model `C = R·K·r·s0` ke hisab se aadha budget ~1% accuracy ke liye kharch hua — ye seedha budget-allocation argument ko support karta hai, aur ye sawal uthata hai ke `R` bhi variable hona chahiye ya nahi.
2. **Round 3 ka dip data volume ki wajah se tha, bug nahi.** Har round ke selected clients ke shard sizes jama karein: round 3 = 2,903 samples (acc 0.611, sab se kam), round 4 = 9,791 samples (acc 0.841). Non-IID skew mein **kaun se** K clients chunte hain utna hi matter karta hai jitna kitne — yehi tension GA ko M4 mein solve karni hai.

### Environment jis par chala (results.json ke `env` block se)

`python 3.12.13 · torch 2.10.0+cu128 · transformers 5.0.0 · peft 0.19.1 · datasets 5.0.0 · numpy 2.0.2 · Tesla T4`

Note: ye CLAUDE.md §4 ke 2023-era pins (transformers 4.36 / peft 0.7) se aage hain. Kaggle image par woh pins force karna numpy 2.x ko todta hai (~20 packages), is liye har run apna exact environment `results.json` ke `env` block mein record karta hai.

### Session 2 status

| File | Kya hai |
|---|---|
| `src/data/glue_loader.py` | SST-2 load + tokenize; eval split = GLUE `validation` (official `test` ke labels −1 hain) |
| `src/fl/dirichlet.py` | label-wise Dirichlet split + empty-client redistribution + `summarize_split` |
| `src/models/lora_wrap.py` | `apply_lora` / `build_lora_model` (rank ek asal argument — M3 ready), `adapter_size_mb` = S |
| `src/fl/client.py` | `local_train` (AdamW, sirf adapters) + `evaluate` |
| `src/fl/simulation.py` | `fedavg_aggregate`, `select_clients`, `run_round`, `run_federated` (resume ke saath) |
| `src/fl/server.py` | CLI: `python -m src.fl.server --config configs/m2_baseline.yaml K=10 R=10` |
| `src/utils/metrics.py`, `checkpoint.py` | accuracy, comm cost (2·K·S), per-round checkpoint/resume |
| `configs/m2_baseline.yaml` | saare hyperparams (koi hard-code nahi) |
| `notebooks/01_m2_baseline.ipynb` | bootstrap → split plot → S → smoke (R=2) → full (R=10) → plots → results push |

**Local env note:** purana `venv/` toota hua tha (repo move + `pip` ka hard-coded path, aur sirf pytest installed). Ab conda env `lorac` (Python 3.11) use karein:
```bash
conda activate lorac      # ya: /Users/muhammaduzair/miniforge3/envs/lorac/bin/python
python -m pytest tests/ -q
```
`numpy` ko `<2` pin kiya gaya hai — torch 2.2.0 wheels numpy 1.x ke against bane hain, numpy 2.x par crash karte hain.

## Architecture Note (sir se confirm)

Ye file **Rasta B** (apna FL loop) follow karti hai — FederatedScope sirf reference. Wajah: M1 mein dekha ke FS ke 2023 pins naye environment se conflict karte hain. Agar sir Rasta A kahein, mujhe (Claude chat) batayein, alternate prompt bana doonga.

---

## Part A: Aap Khud (5 min)

```bash
cd ~/Documents/thesis_LoRaC_GA   # apna path
source venv/bin/activate
git pull
claude
```

---

## Part B: Claude Code Ko Ye Prompt Dein

```
Session 2 — M2: Federated LoRA baseline. Read CLAUDE.md first; confirm scope.

DECISION MADE: We implement our OWN lightweight FedAvg simulation loop
(Route B). third_party/FederatedScope is READ-ONLY reference — do not import
from it in our runtime code. Thesis will cite it as architectural reference.

CONSTRAINTS (unchanged): local = code + CPU unit tests only; training on
Kaggle. Seed 42 everywhere. PEP 8, type hints, Google docstrings. One task
at a time; wait for "next"; show files before writing.

TASK 1 — Data: src/data/glue_loader.py
- load_sst2(tokenizer_name, max_len=128) -> tokenized HF DatasetDict
- Use datasets library; cache-friendly; small helper to subsample per-client
  data to `max_samples_per_client` (config-driven) for cheap runs.
Unit test with a TINY mocked dataset (no download in tests — mock or skip
with @pytest.mark.skipif no network).

TASK 2 — Split: src/fl/dirichlet.py
- dirichlet_split(labels, num_clients=100, alpha=0.3, seed=42) -> list[list[int]]
- Handle edge case: clients with 0 samples -> redistribute (document how).
- summarize_split(...) -> per-client counts + label histogram (for the thesis).
Unit tests: total indices preserved, no overlap, seed reproducibility,
zero-sample handling.

TASK 3 — Model: src/models/lora_wrap.py
- build_lora_model(model_name="distilbert-base-uncased", num_labels=2,
  r=8, alpha=16, dropout=0.05) using peft LoraConfig (target attention
  q_lin/v_lin for DistilBERT).
  Expose `r` (rank) as a real function argument (not hard-coded), so the same
  wrapper supports any rank. Add a unit test that builds the model at r=4 and
  r=16 and confirms adapter_size_mb scales with r.
- report_trainable(model) -> dict(total, trainable, pct)
- adapter_size_mb(model) -> float  # THIS IS OUR S — count only trainable
  params * dtype bytes / 1e6
- get_adapter_state(model) / set_adapter_state(model, state): only LoRA
  params, as CPU tensors (this is what "travels" in simulation).
Unit tests on a tiny config (patch model_name to a small random model or
use the real one guarded by a skip-if-offline marker).

TASK 4 — FL core: src/fl/simulation.py (~150 lines target)
- fedavg_aggregate(states: list[dict], weights: list[int]) -> dict
  (weighted average by client sample count)
- run_round(model, client_loaders, selected_ids, cfg) -> new adapter state
  (sequential local training, 1 local epoch, AdamW lr from config)
- run_federated(cfg) -> history dict {round, test_acc, comm_mb_cumulative}
  comm per round = 2 * K * S (down + up), log it — thesis needs this.
- evaluate(model, test_loader) -> accuracy
- Checkpoint hook: save adapter state + history JSON every round to
  cfg.output_dir; resume-from-checkpoint support.
Unit test: fedavg_aggregate math on toy tensors; run_round on a mocked
2-client tiny setup (CPU).

TASK 5 — Config: configs/m2_baseline.yaml
model, r, alpha, K=10, R=10, local_epochs=1, batch_size=16, lr=2e-4,
max_samples_per_client (default 100 for cheap runs; null = full),
alpha_dirichlet=0.3, num_clients=100, seed=42, output_dir=results/m2_baseline

TASK 6 — Notebook: notebooks/01_m2_baseline.ipynb
Cells: bootstrap (same order as M1 final), load config, build split summary
(print + save plot), build model (print trainable % and S), SHORT SMOKE RUN
(R=2, max_samples_per_client=50) then FULL RUN (R=10), plot accuracy curve
+ cumulative comm, save all to results/m2_baseline/, git push results cell
(token via Kaggle Secrets — placeholder).

TASK 7 — Verify + commit
pytest -v; propose commit message + git commands (I run them).
```

---

## Part C: Kaggle Run

1. Push hone ke baad Kaggle par fresh clone (ya `git pull`)
2. **Pehle SMOKE RUN** (R=2, 50 samples/client) — ~10–15 min. Ye pass ho to hi full run.
3. Full run (R=10) — expect **1–3 ghante** T4 par. Checkpoint har round save hota hai; disconnect ho to resume.
4. Results push → local pull → plot dekhein → sir ko bhejein.

## Kaggle Gotchas (M2 mein asal mein pesh aaye)

| Problem | Hal |
|---|---|
| `Torch not compiled with CUDA enabled` | P100 accelerator select tha (sm_60); Kaggle ka torch sm_70+ chahta hai. **GPU T4 x2** chunein. |
| pip ne numpy 1.26.4 par downgrade kar ke ~20 packages tore | `requirements-kaggle.txt` mein sirf `>=` floors rakhein, kabhi `==` ya upper bound nahi. `numpy<2` sirf **local** constraint hai (torch 2.2.0). |
| `ImportError: incompatible version of torchao (0.10.0)` | peft ≥0.19 ka LoRA dispatcher raise karta hai. Code mein hal ho chuka: `disable_incompatible_torchao_dispatch()` (`src/models/lora_wrap.py`). |
| `ModuleNotFoundError: No module named 'src'` | `!` cell repo directory mein nahi tha. `!cd /kaggle/working/Lorac-GA && python -m ...` ek hi line mein chalayein. |
| Run bina kuch kiye khatam, purani accuracy print | `output_dir` mein pehle se R rounds ke checkpoints maujood the → resume ne loop skip kar diya. Naya `output_dir` dein ya directory delete karein. |
| `ModuleNotFoundError: No module named 'tests.conftest'` | `tests/` package nahi hai; cross-module import hata kar `dataset_factory` fixture use hoti hai. |

## Common Problems

| Problem | Hal |
|---|---|
| CUDA OOM | batch 16→8, ya grad accumulation 2; DistilBERT par unlikely |
| Accuracy ~50% stuck | lr check (2e-4), labels shuffle bug, ya aggregation weights ghalat — TASK 4 ka unit test dobara dekhein |
| Session disconnect | resume-from-checkpoint chalayein — isi liye banaya hai |
| Download slow (SST-2) | HF cache Kaggle par persist nahi hota har session — normal, ~1 min |
| Har client par model copy OOM | model EK hi rahe; sirf adapter states swap hon (TASK 3 ka design yehi hai) |
