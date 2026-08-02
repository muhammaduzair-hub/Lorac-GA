# M4 — GA Integration & K* Selection — Session 4

> **Goal ek line mein:** M1 ke stubs (operators.py, search.py) mein asal GA logic bhar kar, M3 ke A(K) profile par, har budget B ke liye optimal K* nikalna — **aap ki GLUE version of base paper's Table 3.** RQ1 ka doosra adha hissa.

## Definition of Done

- [ ] GA operators implemented + unit-tested: tournament, crossover, mutation, elitism
- [ ] `ga_search()` full loop: P=20, G=30, pc=0.5, pm=0.2, elitism 10% — [1] ke mutabiq
- [ ] A(K) interpolation: GA kisi bhi integer K (1–100) ko evaluate kar sake (profile sirf 8 points hai)
- [ ] 6 budgets B ∈ {10, 50, 100, 400, 1000, 4000} MB par K* table: {B, K*, A(K*), C(K*), f(K*)}
- [ ] Fitness convergence plot (generations vs best f) — base paper Fig 3 ka analog
- [ ] **Sanity vs brute force:** K sirf 1–100 hai — exhaustive check seconds mein ho jata hai; GA ka K* brute-force optimum se match (ya ±1) hona chahiye. Match nahi = GA bug.
- [ ] Sab pushed. **GPU zaroorat: taqreeban zero** — GA profiled curve par chalta hai (P×G=600 lookups, seconds).

## Ek Honest Baat (defense-ready)

Examiner pooch sakta hai: "K sirf 100 values hai, GA kyun, brute force kyun nahi?" Jawab (thesis mein bhi likhein): brute force yahan feasible hai aur hum usay **verification** ke liye use karte hain; GA base paper [1] ki methodology follow karta hai aur bare Kmax / multi-variable extensions par scale karta hai. Ye framing weakness ko strength banati hai.

---

## Part A: Aap Khud

```bash
cd ~/Documents/thesis_LoRaC_GA && source venv/bin/activate && git pull && claude
```

---

## Part B: Claude Code Prompt

```
Session 4 — M4: implement the GA. Read CLAUDE.md. fitness.py already done
(M1). Fill the M1 stubs now; keep signatures compatible with tests.

TASK 1 — src/ga/profile_io.py
- load_profile(json_path) -> dict[int, float]  # K -> acc_mean
- interp_A(profile, K:int) -> float  # linear interpolation between profiled
  K points; clamp at ends. Unit tests incl. exact points and midpoints.

TASK 2 — src/ga/operators.py (fill stubs)
- tournament_selection(pop, fitnesses, size=3, rng) -> Chromosome
- crossover(p1, p2, pc=0.5, rng) -> child K (blend: random of the two, or
  rounded mean — pick one, document)
- mutation(K, pm=0.2, K_max, rng) -> K (random reinit within [1, K_max])
- elitism(pop, fitnesses, ratio=0.1) -> elites list
All take an explicit numpy rng for reproducibility. Unit tests: bounds,
determinism with fixed rng, elitism picks true top.

TASK 3 — src/ga/search.py (fill stub)
- ga_search(profile, B, R, S, K_max=100, P=20, G=30, pc=0.5, pm=0.2,
  elitism_ratio=0.1, seed=42) -> dict{K_star, f_star, history:[best_f per gen]}
- Uses fitness() from M1 + interp_A. Feasibility: skip/penalize K with
  C(K) > B.
- brute_force_K(profile, B, R, S, K_max) -> (K_opt, f_opt)  # verification
Unit tests: on a synthetic concave profile, ga K* == brute force K* (±1);
history is non-decreasing.

TASK 4 — scripts/run_m4.py (CPU, local-runnable!)
Load results/m3_profiles/A_K_profile.json + S from results/m2_baseline
metadata; run ga_search for all six budgets; write
results/m4_ga/K_star_table.json + markdown table + convergence plot
(one line per budget) via src/utils/plots.py.

TASK 5 — pytest -v; run scripts/run_m4.py LOCALLY (no GPU needed) and show
me the K* table; commit message + git commands (I run).
```

---

## Part C: Verify (GPU optional)

1. `python scripts/run_m4.py` **local par hi chal jayega** — GA ko GPU nahi chahiye.
2. Table dekhein: chhote B par chhota K*, bare B par bara K*, saturation ke baad K* aur na barhe — yehi base paper ka pattern [1] hai. Ye pattern GLUE par dikha to **RQ1 ka jawab "haan" ki taraf** hai.
3. Optional GPU verification (recommended, ~1–2 hr): 2–3 chune hue K* par full R=10 run karke confirm karein ke profiled A(K*) asal accuracy ke qareeb hai.
4. Push + sir ko table bhejein — ye aap ka pehla "headline result" hai.

## Common Problems

| Problem | Hal |
|---|---|
| GA != brute force | operators ke bounds/rng bugs — TASK 2 tests dobara; pm barha kar dekhein |
| Sab budgets par same K* | S ghalat (bohot chhota) — M2 ka adapter_size_mb dobara check |
| f(K*) hamesha B/C side par | budgets ke units MB confirm karein; R=10 hi hai? |
| History decrease karti hai | elitism bug — elites ko unchanged carry karein |
