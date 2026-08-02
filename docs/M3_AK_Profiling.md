# M3 — A(K) Profiling — Session 3

> **Goal ek line mein:** Mukhtalif K values par short federated runs chala kar accuracy-vs-K curve `A(K)` banana — ye GA (M4) ka empirical input hai. **RQ1 ka pehla adha hissa.**

## Definition of Done

- [ ] K ∈ {2, 5, 10, 20, 40, 60, 80, 100} par runs complete (har K par R=10)
- [ ] Har K **2 seeds** se (42, 43) — mean ± std report (ek seed = examiner ka easy target)
- [ ] `results/m3_profiles/A_K_profile.json` — schema: {K, seeds:[...], acc_mean, acc_std, comm_mb}
- [ ] Accuracy-vs-K plot (thesis Figure banega) — diminishing returns dikhna chahiye
- [ ] Sanity: A(K) generally barhta hua, phir saturate — agar random/ulta hai to bug hai
- [ ] Sab GitHub par pushed

## GPU Budget Warning (pehle parh lein)

8 K-values × 2 seeds × 10 rounds = **16 runs**. Bare K (80, 100) mehngi runs hain (zyada clients = zyada local trainings per round). Andaza: **total 12–20 GPU ghante.** Kaggle ka 30 hr/week quota — **2 hafton mein baantein**: hafta 1 mein K ≤ 20, hafta 2 mein K ≥ 40. `max_samples_per_client` cap (jaise 100) use karein warna quota khatam.

**Honest trade-off note (thesis mein likhna):** capped samples + R=10 se A(K) ka *shape* milta hai (jo GA ko chahiye), absolute best accuracy nahi. Ye base paper ke "profiled via short runs" approach ke ain mutabiq hai [1].

---

## Part A: Aap Khud

```bash
cd ~/Documents/thesis_LoRaC_GA && source venv/bin/activate && git pull && claude
```

---

## Part B: Claude Code Prompt

```
Session 3 — M3: A(K) profiling. Read CLAUDE.md. M2 code (simulation.py,
glue_loader, dirichlet, lora_wrap) is the foundation — reuse, don't rewrite.

TASK 1 — src/fl/profiler.py
- profile_AK(cfg, K_values: list[int], seeds: list[int]) -> dict
  For each (K, seed): call run_federated with that K, collect final test
  accuracy + total comm MB. Aggregate mean/std per K.
- Incremental saving: after EVERY (K, seed) run, update
  results/m3_profiles/A_K_profile.json (crash = kuch nahi khota).
- resume support: skip (K, seed) pairs already in the JSON.
Unit test: mock run_federated -> verify aggregation, resume-skip logic,
JSON schema.

TASK 2 — configs/m3_profile.yaml
K_values: [2, 5, 10, 20, 40, 60, 80, 100]; seeds: [42, 43]; R=10;
max_samples_per_client: 100; inherit rest from m2_baseline.

TASK 3 — src/utils/plots.py
- plot_AK(profile_json, out_pdf): accuracy vs K with error bars (std),
  seaborn style, 300 DPI PDF (thesis-quality), axis labels with units.
Unit test with a fake profile dict.

TASK 4 — notebooks/02_m3_profile.ipynb
bootstrap cells -> load config -> OPTION to run subset (K_subset param, for
quota splitting across weeks) -> profiler run -> plot -> push results.

TASK 5 — pytest -v; commit message + git commands (I run).
```

---

## Part C: Kaggle (2 hafte ka plan)

- **Hafta 1:** notebook mein `K_subset=[2,5,10,20]` → run → push
- **Hafta 2:** `K_subset=[40,60,80,100]` → run → push (resume JSON merge kar lega)
- Har hafte ke baad plot dekhein: curve barh kar flatten ho rahi hai? Perfect.

## Common Problems

| Problem | Hal |
|---|---|
| Quota khatam beech mein | incremental JSON hai — agle hafte wahin se resume |
| K=100 bohot slow | max_samples_per_client 100→50 sirf bare K ke liye (document karein) |
| A(K) non-monotonic thora sa | 2 seeds ka std dekhein — chhota noise normal hai; bara ulta trend = bug |
| Curve bilkul flat | non-IID split verify karein (M2 ka summarize_split) — IID split par K ka farq kam dikhta hai |
