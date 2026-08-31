# M3 — A(K, r) Profiling — Session 3

> **Goal ek line mein:** Mukhtalif (K, r) jorron par short federated runs chala kar accuracy surface `A(K, r)` banana — ye joint-optimization GA (M4) ka empirical input hai. RQ1 (A(K) characterization) ka data aur RQ2 (K–r frontier) ka setup yahin banta hai.

Cost formula: `C = R · K · r · s0`, jahan s0 per-unit-rank payload (MB) hai. Isi wajah se profiling do variables par hoti hai — client count K aur rank r.

## Definition of Done

- [ ] K ∈ {2, 5, 10, 20, 40, 80} × r ∈ {2, 4, 8, 16} par runs — coarse grid (24 cells)
- [ ] Har cell 1 seed pehle (42); sirf frontier ke qareeb wale cells 2nd seed (43) se
- [ ] `results/m3_profiles/A_Kr_surface.json` — schema: {K, r, seed, acc, comm_mb, s0_per_rank}
- [ ] Do plots: (a) A(K) at fixed r=8 → RQ1 curve; (b) A(K, r) heatmap → RQ2 ka setup
- [ ] Sanity RQ1: r=8 wali row par accuracy K ke saath barhe phir saturate ho
- [ ] Sanity RQ2: har K par r barhne se accuracy barhe (kabhi plateau) — flat ho to note karein
- [ ] `s0` (per-unit-rank payload MB) documented — M4 ke cost formula ka basis
- [ ] Sab GitHub par pushed

## GPU Budget Warning (pehle zaroor parhein)

6 K × 4 r = 24 cells (+ kuch 2nd-seed) ≈ 28–32 runs. Bare cells (K=80, r=16) mehngi hain. Andaza: total ~30–45 GPU ghante. Kaggle 30 hr/week — 2–3 hafton mein baantein. Do bachat techniques (zaroori):

1. **`max_samples_per_client` cap** (jaise 100) — surface ka *shape* chahiye, absolute SOTA accuracy nahi. Base paper bhi short runs se profile karta hai [1].
2. **Grid ko coarse rakhein.** 6×4 kafi hai frontier dikhane ke liye. Fine grid ki zaroorat nahi — GA bilinear interpolation kar lega (M4).

**Honest thesis note (likhna):** "A(K, r) was profiled on a coarse grid with capped local data to recover the accuracy structure; absolute accuracy would improve with full data and more rounds, but the relative (K, r) trade-off — which drives the optimization — is preserved."

## Grid Strategy (smart ordering)

Runs is tarteeb se chalayein taa ke aadha quota khatam ho bhi to useful data ho:

```
Priority 1 (RQ1 curve):     r=8 fixed, all K        → 6 runs  (hafta 1)
Priority 2 (RQ2 frontier):  K∈{5,10,20}, all r      → 9 runs  (hafta 1-2)
Priority 3 (corners):       remaining cells          → ~9 runs (hafta 2-3)
Priority 4 (2nd seed):      frontier cells only      → ~6 runs (hafta 3)
```

---

## Part A: Aap Khud

```bash
cd ~/Documents/thesis_LoRaC_GA && source venv/bin/activate && git pull && claude
```

---

## Part B: Claude Code Prompt

```
Session 3 — M3: A(K, r) surface profiling (two-variable). Read CLAUDE.md.
M2 code (simulation.py, glue_loader, dirichlet, lora_wrap) is the foundation
— reuse. lora_wrap accepts a `rank` argument (from M2), so sweeping rank is
straightforward.

TASK 1 — src/fl/profiler.py
- profile_AKr(cfg, K_values, r_values, seeds, priority_cells=None) -> dict
  For each (K, r, seed): build the LoRA model at that rank, run run_federated
  with that K, collect final test accuracy + total comm MB + measured
  s0 (per-unit-rank payload = adapter_size_mb / r). Aggregate per (K, r).
- Incremental saving: after EVERY (K, r, seed) run, update
  results/m3_profiles/A_Kr_surface.json (crash = nothing lost).
- resume support: skip (K, r, seed) triples already present.
- priority_cells: optional ordered list of (K, r) so the RQ1 curve and RQ2
  frontier run first (quota may run out).
Unit test: mock run_federated -> verify aggregation over seeds, resume-skip,
JSON schema includes K, r, seed, acc, comm_mb, s0.

TASK 2 — configs/m3_profile.yaml
K_values: [2, 5, 10, 20, 40, 80]
r_values: [2, 4, 8, 16]
seeds: [42]            # 2nd seed [43] only for frontier cells, via priority
R: 10
max_samples_per_client: 100
priority_cells: [[2,8],[5,8],[10,8],[20,8],[40,8],[80,8],   # RQ1 curve (r=8)
                 [5,2],[5,4],[5,16],[10,2],[10,4],[10,16],[20,2],[20,4],[20,16]]
inherit rest from m2_baseline.

TASK 3 — src/utils/plots.py  (add two functions)
- plot_AK_curve(surface_json, r_fixed=8, out_pdf): accuracy vs K at fixed r,
  error bars if 2 seeds — this is the RQ1 figure.
- plot_AKr_heatmap(surface_json, out_pdf): K (x) vs r (y) grid, cell colour =
  accuracy, annotate values — this is the RQ2 setup figure. seaborn, 300 DPI.
Unit tests with a small fake surface dict.

TASK 4 — notebooks/02_m3_profile.ipynb
bootstrap cells -> load config -> run profiler with priority ordering and an
optional `cell_subset` param (for splitting across weeks) -> both plots ->
push results. Print a progress line per completed cell (K, r -> acc).

TASK 5 — pytest -v; commit message + git commands (I run).
```

---

## Part C: Kaggle (2–3 hafte ka plan)

- **Hafta 1:** `cell_subset` = RQ1 curve (r=8, all K) + a few frontier cells → run → push. RQ1 ka jawab (curve) mojood.
- **Hafta 2:** baqi frontier cells (K∈{5,10,20} × all r) → run → push (resume merge karega).
- **Hafta 3:** corners + frontier ka 2nd seed → run → push.
- Har hafte plots dekhein: curve saturate ho rahi? heatmap mein r ka gradient dikh raha?

## Common Problems

| Problem | Hal |
|---|---|
| Quota khatam beech mein | incremental JSON hai — priority ordering se aham data pehle aa jata hai |
| r=16, K=80 bohot slow | us corner ko last/skip rakhein; GA interpolation se kaam chal jata hai |
| A(K,r) mein r ka farq flat | LoRA rank chhote model par kam matter karta hai — RoBERTa-base try karein, ya likh dein "rank saturates early" (yeh bhi ek RQ2 finding hai) |
| s0 har r par thora alag | expected — adapter size r ke saath bilkul linear nahi; per-r store karein (json mein already hai) ya mean use karein |
| heatmap non-monotonic patches | 1-seed noise — frontier cells ka 2nd seed isi liye add karte hain |
