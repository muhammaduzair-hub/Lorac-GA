# Thesis Milestones — Master Overview

> LoRaC-GA on Real LLMs — 6 milestones, har ek ki apni file. Har file ka format: **Goal → Definition of Done → Part A (aap khud) → Part B (Claude Code prompt) → Part C (Kaggle/push) → Common Problems.**

Thesis budget-aware joint optimization karti hai: ek fixed uplink budget ke andar client count K aur LoRA rank r dono ko ek saath choose karna. Cost `C = R · K · r · s0`.

## Roadmap

| # | Milestone | Hafte | Status | File |
|---|-----------|-------|--------|------|
| M1 | Environment & Pipeline Setup | 1–2 | ✅ COMPLETE | M1_Setup_Pipeline.md |
| M2 | Federated LoRA Baseline (FedAvg, fixed K) | 3–5 | ✅ COMPLETE — 85.21%, S=2.9583 MB, 591.7 MB | M2_Federated_Baseline.md |
| M3 | A(K, r) Surface Profiling | 6–8 | ⬜ | M3_AK_Profiling.md |
| M4 | GA Integration & Joint (K*, r*) Selection | 9–10 | ⬜ | M4_GA_Integration.md |
| M5 | Baseline Comparison & Full Evaluation | 11–13 | ⬜ | M5_Baseline_Comparison.md |
| M6 | Analysis & Thesis Writing | 14–16+ | ⬜ | M6_Analysis_Writing.md |

## RQ Mapping (defense ke liye yaad rakhein)

- **M1–M2** = groundwork (pipeline + baseline)
- **M3** = **RQ1** — characterization: kya A(K) real LLM par saturating structure dikhati hai?
- **M4** = **RQ2** — joint budget allocation: optimal (K*, r*), fixed-r par gain ← contribution
- **M5** = **RQ3** — method benefit: joint (K,r) vs fixed-K / random-K / FedAvg / FedSA-LoRA
- **M6** = consolidation

## Key Formulas aur Symbols

- Cost: `C = R · K · r · s0` (s0 = per-unit-rank payload MB, M3 mein measure hota hai)
- Fitness: `f(K, r) = min(A(K, r), B / (R·K·r·s0))`
- K = client count, r = LoRA rank, R = rounds, B = uplink budget
- M2 ka `lora_wrap` `rank` argument leta hai, taa ke M3 rank sweep kar sake

## Golden Rules (har milestone par lagu)

1. **Local = code + unit tests. Kaggle = training.** FederatedScope locally install nahi karna.
2. **Har session = commit + push.** Kaam kabhi sirf Kaggle session mein na chhorein.
3. **Seed 42 everywhere** — torch, numpy, random. Har experiment reproducible.
4. **Chhote results commit karein** (JSON/CSV/plots), bare `*.pt` checkpoints nahi.
5. **Sir ko Friday update** — `git log --oneline` + ek line "kya hua, kya stuck, agla kya".
6. **Ek waqt mein ek Claude Code task.** "next" bol kar aage barhein, har file approve karein.
7. **Warning vs Error triage:** version-conflict warning = ignore; ImportError/traceback = asal masla.

## Architecture Decision (M2 se pehle sir se confirm)

**Rasta B (recommended, in files ka default):** Apna ~150-line FedAvg simulation loop (PEFT + transformers directly). FederatedScope repo sirf reference — kyunke uske 2023 pins naye Kaggle environment se conflict karte hain.
**Rasta A (alternative):** FederatedScope ka runner use karna — agar sir insist karein.

Thesis wording (Rasta B): *"Our simulation is implemented directly with HuggingFace transformers and PEFT, informed by the FederatedScope-LLM [7] architecture."*

## Kaggle Quota Planning

- Free tier: ~30 GPU hrs/week, weekend reset.
- **M2:** ~2–4 hrs total (chhoti runs se shuru).
- **M3:** sab se bhaari — 2-D surface (~24 cells) = ~30–45 GPU hrs; 2–3 hafton mein baant lein.
- **M4:** GA khud seconds mein (profiled surface par); GPU sirf 2–3 (K*,r*) verification runs ke liye (~1–2 hrs optional).
- **M5:** 5 methods × 3 budgets × 2 seeds = ~10–15 hrs; 1–2 hafton mein.

## Timeline Reality Check

10 hrs/week × 16 hafte = 160 hrs budget. Agar time tight ho to is tarteeb se trim karein (upar wala pehle cut):
1. M5 baselines 4 → 2 (fixed-K + FedSA-LoRA)
2. Budgets 3 → 2 ya grid 6×4 → 4×3
3. Sirf 1 model (RoBERTa recommended — rank par zyada responsive)
4. SST-2 drop, sirf MNLI

Contribution (RQ2 joint K,r — M4) hamesha protect karein; baqi negotiable.
