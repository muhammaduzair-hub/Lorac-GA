# M4 — GA Integration & Joint (K*, r*) Selection — Session 4

> **Goal ek line mein:** M1 ke stubs mein asal GA logic bhar kar, M3 ke A(K, r) surface par, har budget B ke liye optimal `(K*, r*)` jori nikalna — aur ye saabit karna ke joint optimization fixed-r K-only se behtar hai. Ye RQ2 (joint allocation) ka core contribution deliver karta hai.

Fitness: `f(K, r) = min(A(K, r), B / (R·K·r·s0))`. GA iso-budget frontier `K·r = const` par best jori dhoondta hai. GA vs exhaustive ek methodology verification hai (grid chhota hai, to exact optimum se match confirm hota hai) — research question nahi. Baselines se comparison M5 mein hai.

## Definition of Done

- [ ] GA operators (K aur r dono ko encode karte hue): tournament, crossover, mutation, elitism — unit-tested
- [ ] `ga_search()` full loop over (K, r) space: P=20, G=30, pc=0.5, pm=0.2, elitism 10%
- [ ] A(K, r) 2-D bilinear interpolation — GA kisi bhi (K, r) ko evaluate kar sake
- [ ] 6 budgets B ∈ {10, 50, 100, 400, 1000, 4000} MB par (K*, r*) table: {B, K*, r*, A(K*,r*), C, f}
- [ ] RQ2 headline number: har budget par joint-(K,r) accuracy vs fixed-r=8 K-only accuracy — "joint gains +X%"
- [ ] Methodology check: GA ka (K*, r*) exhaustive search ke optimum se match kare (ya adjacent cell). Grid chhota (6×4=24) → exhaustive milliseconds mein.
- [ ] Fitness convergence plot + iso-budget frontier plot
- [ ] Sab pushed. GPU zaroorat: taqreeban zero — GA profiled surface par chalta hai.

## RQ2 — Ye Milestone Ka Asal Contribution

Sirf (K*, r*) table kaafi nahi — comparison chahiye. Har budget par teen cheezein compute karein:
1. Joint optimum: `max over (K,r) of f(K,r)`
2. Fixed-r baseline: `max over K of f(K, r=8)` (r fixed, jaise base paper)
3. Gap: joint accuracy − fixed-r accuracy → yehi aap ka RQ2 contribution number hai

## GA vs Exhaustive — Methodology Verification

Grid chhota hai, isliye GA ke natije ko exhaustive search se verify karein — dono same (K*, r*) dene chahiye, jo confirm karta hai ke GA sahi optimize kar raha hai. GA is tarah ke bare search spaces (finer r, larger Kmax) ke liye chosen hai jahan exhaustive mehnga ho jata hai. Ye ek do-line note hai, poora research question nahi.

---

## Part A: Aap Khud

```bash
cd ~/Documents/thesis_LoRaC_GA && source venv/bin/activate && git pull && claude
```

---

## Part B: Claude Code Prompt

```
Session 4 — M4: joint (K, r) optimization. Read CLAUDE.md. fitness.py exists
(M1); extend it to two variables this session.

TASK 1 — src/ga/fitness.py  (two-variable)
- comm_cost(K, r, R, s0) -> float           # R*K*r*s0
- efficiency(K, r, B, R, s0) -> float        # B / (R*K*r*s0); guard zero
- fitness(K, r, A_Kr, B, R, s0) -> float     # min(A_Kr, efficiency(...))
Keep single-var signatures as thin wrappers (r=r_fixed) so M1 tests still
pass; add two-var tests.

TASK 2 — src/ga/profile_io.py
- load_surface(json) -> dict[(K,r)] = acc_mean
- interp_A(surface, K, r) -> float   # BILINEAR interpolation over the (K,r)
  grid; clamp at edges. Unit tests: exact grid points, midpoints, corners.

TASK 3 — src/ga/operators.py  (fill stubs; chromosome = (K, r))
- Chromosome dataclass: K:int, r:int, fitness:float|None
- tournament_selection, crossover (blend both K and r), mutation (reinit K
  and/or r within bounds), elitism. Explicit numpy rng. Unit tests: bounds on
  both dims, determinism, elitism correctness.

TASK 4 — src/ga/search.py
- ga_search(surface, B, R, s0, K_max, r_values, P=20, G=30, pc, pm,
  elitism_ratio, seed) -> {K_star, r_star, f_star, history}
  Uses fitness() + interp_A. Feasibility: penalize/skip (K,r) with C>B.
- brute_force(surface, B, R, s0, K_max, r_values) -> (K,r,f)  # exhaustive,
  used to verify ga_search is correct.
Unit tests: on a synthetic concave surface, GA optimum == brute force (±1
cell).

TASK 5 — scripts/run_m4.py  (CPU, local!)
Load results/m3_profiles/A_Kr_surface.json + s0. For each budget B in the six
levels:
 (a) joint optimum via GA (assert it equals brute_force result — verification),
 (b) fixed-r baseline: best K at r=8 only,
 (c) record gap = joint_acc - fixedr_acc.
Write results/m4_ga/:
 - Kr_star_table.json + markdown  (RQ2 table with gap column)
 - ga_verification.json  (one line per budget: ga==brute_force? yes/no)
 - convergence plot; iso-budget frontier plot (K·r=const lines with optima).

TASK 6 — pytest -v; run scripts/run_m4.py LOCALLY (no GPU) and show me the
RQ2 (K*,r*)-with-gap table + the verification file; commit message + git
commands (I run).
```

---

## Part C: Verify (GPU optional)

1. `python scripts/run_m4.py` local par chalega — GA ko GPU nahi chahiye.
2. **RQ2 table dekhein:** kya joint (K*, r*) fixed-r=8 se behtar hai? Kitna? Ye aap ka headline contribution number hai. (Kuch budgets par gap ~0 ho sakta hai — bhi valid: "at these budgets r=8 is already near-optimal".)
3. **Verification file dekhein:** har budget par GA == exhaustive? Sab "yes" hone chahiye.
4. Optional GPU verification (~1–2 hr): 2–3 chune hue (K*, r*) par full run karke confirm karein ke profiled A(K*,r*) asal accuracy ke qareeb hai.
5. Push + sir ko RQ2 table bhejein — gap number aap ka pehla contribution result hai.

## Common Problems

| Problem | Hal |
|---|---|
| Joint gap hamesha ~0 | Chhote model par r=8 already saturated — RoBERTa-base ya chhote r {2,4} par dekhein; ya honestly report karein (RQ2 ka jawab) |
| GA != brute force | operators bounds/rng bug (do dimensions) — TASK 3 tests dobara; mutation dono dims touch kare |
| interp_A edge par NaN | bilinear clamp lagayein grid edges par (TASK 2) |
| Sab budgets same (K*,r*) | s0 ghalat, ya budgets range chhota — M3 ka s0 aur units confirm |
| GA vs baselines kahan hai | Wo M5 mein hai (RQ3). M4 sirf (K*,r*) nikalta + GA ko exhaustive se verify karta hai. |
