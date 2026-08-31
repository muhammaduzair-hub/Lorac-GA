# M5 — Baseline Comparison & Full Evaluation — Session 5

> **Goal ek line mein:** Budget-aware joint `(K*, r*)` selection ko standard baselines — fixed-K, random-K, FedAvg/FedIT, aur FedSA-LoRA — ke khilaf, SAME budget par compare karna. Ye RQ3 (method benefit) ka jawab hai.

RQ3 aap ke kaam ko field ke against test karta hai, apne tool ke against nahi. Method-under-test joint (K*, r*) hai (M4 se); baselines mein woh tareeqe hain jo K aur r ko fix ya alag rakhte hain.

## Definition of Done

- [ ] Fair-comparison protocol: har method ko wohi budget B — feasible (K, r) with C = R·K·r·s0 ≤ B
- [ ] Chaar baselines implemented: fixed-K (r=8), random-K (random feasible K,r), FedAvg/FedIT (r=8 matched), FedSA-LoRA (A-matrices only, r=8)
- [ ] Method-under-test = joint (K*, r*) from results/m4_ga/Kr_star_table.json
- [ ] Teen budgets par (compute bachat: {10, 100, 1000} MB), har method, 2 seeds
- [ ] Metrics table: final accuracy, total comm MB, rounds-to-85% (SST-2 target)
- [ ] Plots: (a) accuracy-vs-rounds per budget (methods overlaid); (b) accuracy-vs-comm scatter
- [ ] RQ3 verdict likha: joint (K*,r*) baselines se better/equal/worse — kahan/kyun?
- [ ] Sab pushed. GPU andaza: 5 methods × 3 budgets × 2 seeds × R=10 ≈ 10–15 hrs — 1–2 hafton mein baantein.

## Honest Framing (pehle se soch lein)

Mumkin hai joint (K*, r*) har jagah na jeete. Mixed results bhi valid thesis hain agar aap explain kar sakein *kahan* aur *kyun*. "Joint selection matches accuracy at 40% less communication" bhi jeet hai, chahe raw accuracy barabar ho.

---

## Part A: Aap Khud

```bash
cd ~/Documents/thesis_LoRaC_GA && source venv/bin/activate && git pull && claude
```

---

## Part B: Claude Code Prompt

```
Session 5 — M5: baseline comparison (RQ3). Read CLAUDE.md. Reuse
simulation.py; extend, don't fork. Method-under-test is joint (K*, r*).

TASK 1 — src/fl/baselines.py
- budget_matched_K(B, R, s0, r, K_max) -> int  # largest feasible K under B
  at a given rank r (used by fixed-K / FedAvg / FedSA at r=8)
- random_feasible_Kr(B, R, s0, K_max, r_values, rng) -> (K, r)  # random draw
  under budget (averaged over a few draws for the random baseline)
- fedsa_lora mode: modify get/set adapter state + aggregate to exchange ONLY
  lora_A tensors (lora_B stays local per client, persisted across rounds in a
  client_state dict). Comm per round = 2*K*S_A. Implement as a cfg flag
  `aggregation: fedavg|fedsa` in simulation.py — minimal diff.
Unit tests: budget_matched_K math; random draw stays feasible; fedsa
aggregate touches only A keys; client B-state persistence across rounds.

TASK 2 — src/fl/experiments.py
- run_comparison(cfg) loops over budgets x methods x seeds:
  methods = {
    "joint_kr":      (K*, r*) from results/m4_ga/Kr_star_table.json,  # OURS
    "fixed_k_r8":    budget_matched_K at r=8 (single-variable),
    "random_kr":     random_feasible_Kr under B (avg of a few draws),
    "fedavg_matched":budget_matched_K at r=8,
    "fedsa_matched": budget_matched_K at r=8 with S_A
  }
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

1. Smoke: ek (budget=100, method=joint_kr, seed=42) run pehle — theek to baqi sab.
2. Quota ke hisaab se subset mein chalayein; resume hai hi.
3. Summary table nikalte hi RQ3 par ek paragraph likh dein (M6 mein copy hoga) — results taza hon to analysis behtar.
4. Push + sir ko summary table + dono plots.

## Common Problems

| Problem | Hal |
|---|---|
| FedSA accuracy crash | B-matrix persistence bug — client_state per client verify (TASK 1 test) |
| matched-K > 100 | K_max clamp karein; bare budgets par sab methods K_max par converge — expected, likh dein |
| rounds-to-85% kabhi nahi milta | chhote budgets par normal (K chhota) — table mein ">R" likhein |
| random_kr bohot noisy | 3–5 draws ka average lein; seed fix rakhein |
| Seeds mein bara variance | 3rd seed sirf conflicted cases par add karein |
