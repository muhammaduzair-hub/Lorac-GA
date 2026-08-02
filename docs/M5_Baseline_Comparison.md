# M5 — Baseline Comparison & Full Evaluation — Session 5

> **Goal ek line mein:** LoRaC-GA-style K* selection ko FedAvg/FedIT [2] aur FedSA-LoRA [9] ke khilaf, SAME budget par, teen metrics par compare karna. **RQ2 (preservation) + RQ3 (comparison) ka jawab.**

## Definition of Done

- [ ] Fair-comparison protocol implemented: har baseline ko wohi budget B — FedAvg ke liye K aisa ke C(K)=R·K·S ≤ B (base paper ka matching rule [1])
- [ ] FedSA-LoRA variant: sirf **A-matrices** aggregate (B-matrices local) [9] — humare loop mein ~20-line change; S_A ≈ S/2 (comm bhi aadhi)
- [ ] Teen budgets par (compute bachat: {10, 100, 1000} MB), har method, 2 seeds
- [ ] Metrics table: final accuracy, total comm MB, rounds-to-85% (SST-2 ka reasonable target)
- [ ] Plots: (a) accuracy-vs-rounds per budget (methods overlaid) — base paper Fig 2 analog; (b) accuracy-vs-comm scatter
- [ ] RQ2 verdict likha: savings preserved? RQ3 verdict: GA-K* better/equal/worse aur kahan?
- [ ] Sab pushed. GPU andaza: **3 budgets × 3 methods × 2 seeds × R=10 ≈ 8–12 hrs** — ek hafte ke quota mein mumkin, warna 2 mein baantein.

## Honest Framing (pehle se soch lein)

Mumkin hai GA-K* har jagah na jeete — jaise base paper mein bhi FedProx kuch rounds mein FedAvg se aage tha. **Mixed results bhi valid thesis hain** agar aap explain kar sakein *kahan* aur *kyun*. "GA matches accuracy at 40% less communication" bhi jeet hai, chahe raw accuracy barabar ho.

---

## Part A: Aap Khud

```bash
cd ~/Documents/thesis_LoRaC_GA && source venv/bin/activate && git pull && claude
```

---

## Part B: Claude Code Prompt

```
Session 5 — M5: baseline comparison. Read CLAUDE.md. Reuse simulation.py;
extend, don't fork.

TASK 1 — src/fl/baselines.py
- budget_matched_K(B, R, S, K_max) -> int  # largest feasible K under B
  (this is how FedAvg gets a "fair" K per base paper's matching rule)
- fedsa_lora mode: modify get/set adapter state + aggregate to exchange ONLY
  lora_A tensors (lora_B stays local per client, persisted across rounds in
  a client_state dict). Comm per round = 2*K*S_A. Implement as a cfg flag
  `aggregation: fedavg|fedsa` in simulation.py — minimal diff.
Unit tests: budget_matched_K math; fedsa aggregate touches only A keys;
client B-state persistence across rounds.

TASK 2 — src/fl/experiments.py
- run_comparison(cfg) loops over budgets x methods x seeds:
  methods = {"ga_kstar": K from results/m4_ga/K_star_table.json,
             "fedavg_matched": budget_matched_K,
             "fedsa_matched": budget_matched_K with S_A}
- Per-run outputs -> results/m5_compare/{budget}_{method}_{seed}.json
  (history incl. per-round acc + cumulative comm). Incremental + resume,
  same pattern as M3 profiler.
- summarize() -> results/m5_compare/summary.json + markdown table with
  acc mean±std, total comm, rounds-to-85%.

TASK 3 — configs/m5_compare.yaml
budgets: [10, 100, 1000]; seeds: [42, 43]; target_acc: 0.85; rest inherited.

TASK 4 — plots: extend src/utils/plots.py
- plot_convergence(per-budget, methods overlaid, error bands)
- plot_acc_vs_comm(scatter, one marker per method-budget)
300 DPI PDFs. Unit tests with fake histories.

TASK 5 — notebooks/03_m5_compare.ipynb
bootstrap -> subset option (budget_subset for quota splitting) -> run ->
summarize -> plots -> push results.

TASK 6 — pytest -v; commit message + git commands (I run).
```

---

## Part C: Kaggle + Analysis

1. Smoke: ek (budget=100, method=ga_kstar, seed=42) run pehle — theek to baqi sab.
2. Quota ke hisaab se subset mein chalayein; resume hai hi.
3. Summary table nikalte hi **RQ2/RQ3 par ek paragraph likh dein** (M6 mein copy hoga) — results taza hon to analysis behtar hota hai.
4. Push + sir ko summary table + dono plots.

## Common Problems

| Problem | Hal |
|---|---|
| FedSA accuracy crash | B-matrix persistence bug — client_state per client verify (TASK 1 test) |
| FedAvg matched-K > 100 | K_max clamp karein; bare budgets par sab methods K_max par converge karenge — expected, likh dein |
| rounds-to-85% kabhi nahi milta | chhote budgets par normal (K chhota) — table mein ">R" likhein |
| Seeds mein bara variance | 3rd seed add sirf conflicted cases par |
