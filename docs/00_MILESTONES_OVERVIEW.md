# Thesis Milestones — Master Overview

> LoRaC-GA on Real LLMs — 6 milestones, har ek ki apni file. Har file ka format: **Goal → Definition of Done → Part A (aap khud) → Part B (Claude Code prompt) → Part C (Kaggle/push) → Common Problems.**

## Roadmap

| # | Milestone | Hafte | Status | File |
|---|-----------|-------|--------|------|
| M1 | Environment & Pipeline Setup | 1–2 | ✅ COMPLETE | M1_Setup_Pipeline.md |
| M2 | Federated LoRA Baseline (FedAvg, fixed K) | 3–5 | ⬜ | M2_Federated_Baseline.md |
| M3 | A(K) Profiling | 6–8 | ⬜ | M3_AK_Profiling.md |
| M4 | GA Integration & K* Selection | 9–10 | ⬜ | M4_GA_Integration.md |
| M5 | Baseline Comparison & Full Evaluation | 11–13 | ⬜ | M5_Baseline_Comparison.md |
| M6 | Analysis & Thesis Writing | 14–16+ | ⬜ | M6_Analysis_Writing.md |

## RQ Mapping (defense ke liye yaad rakhein)

- M1–M2 = groundwork
- M3–M4 = **RQ1** (transferability) ka jawab
- M5 = **RQ2** (preservation) + **RQ3** (comparison) ka jawab
- M6 = consolidation

## Golden Rules (har milestone par lagu)

1. **Local = code + unit tests. Kaggle = training.** FederatedScope locally install nahi karna.
2. **Har session = commit + push.** Kaam kabhi sirf Kaggle session mein na chhorein.
3. **Seed 42 everywhere** — torch, numpy, random. Har experiment reproducible.
4. **Chhote results commit karein** (JSON/CSV/plots), bare `*.pt` checkpoints nahi.
5. **Sir ko Friday update** — `git log --oneline` + ek line "kya hua, kya stuck, agla kya".
6. **Ek waqt mein ek Claude Code task.** "next" bol kar aage barhein, har file approve karein.
7. **Warning vs Error triage:** version-conflict warning = ignore; ImportError/traceback = asal masla.

## Architecture Decision (M2 se pehle sir se confirm)

**Rasta B (recommended, in files ka default):** Apna ~150-line FedAvg simulation loop (PEFT + transformers directly). FederatedScope repo sirf **reference** — kyunke uske 2023 pins naye Kaggle environment se conflict karte hain (install mein dekh chuke hain).
**Rasta A (alternative):** FederatedScope ka runner use karna — agar sir insist karein, to M2 file ke Part B mein "RASTA A VARIANT" note dekhein.

Thesis wording (Rasta B): *"Our simulation is implemented directly with HuggingFace transformers and PEFT, informed by the FederatedScope-LLM [7] architecture."*

## Kaggle Quota Planning

- Free tier: ~30 GPU hrs/week, weekend reset.
- M2: ~2–4 hrs total (chhoti runs se shuru).
- M3: **sab se bhaari** — 8 profiling runs; 2 hafton ke quota mein baant lein.
- M4: GA khud seconds mein (profiled A(K) par) — GPU sirf verification runs ke liye.
- M5: ~6–10 hrs (baselines × budgets).
