# M0 — EMNIST Replication of LoRaC-GA (Validation) — Pehle ye, phir LLM

> **Goal ek line mein:** Base paper (Solat & Lee, *Sensors* 2025) ka K-only GA usi dataset (EMNIST Balanced) aur usi setup par dobara chala kar dekhna ke uska *trend* reproduce hota hai — taa ke LLM par transfer (M2+) ek **verified** GA par khada ho. Sir ka faisla: pehle EMNIST, phir LLM.

**Scope note:** Ye *validation milestone* hai, thesis ka contribution nahi. EMNIST sirf yahan; M2–M6 SST-2/LLM par rahenge. K-only (rank r nahi — paper ka fitness bhi K-only hai).

---

## Paper ka Asal Setup (Section 5.1, Table 2)

| Item | Paper ki value |
|------|----------------|
| Dataset | EMNIST **Balanced** (131,600 samples, 47 classes) |
| Clients / split | `K_max = 100`, Dirichlet label split, `α = 0.3`, non-overlapping |
| Model | "compact transformer-based classifier", LoRA "selected linear layers" |
| Adapter payload | `S = 0.0833 MB` (paper ke mutabiq FeDeRA [5] se liya) |
| Rounds | `R = 10` |
| A(K) | Empirical, "training curves se profiled" |
| GA | `P=20, G=30, p_c=0.5, p_m=0.2, ρ=10% elitism, tournament size 3` |
| Budgets B | {10, 50, 100, 400, 1000, 4000} MB |
| Baselines | FedAvg, FedProx (µ=0.01), CL (centralized); fitness plot: GA vs greedy vs random |
| Fitness | `f(K) = min(A(K), B / (R·K·S))` |

Existing repo defaults (CLAUDE.md §6) paper se match karte hain: P=20, G=30, p_c=0.5, p_m=0.2, α=0.3, R=10.

## Jo Paper Nahi Batata (guess nahi karna — config mein "assumption" likhna)

1. Model architecture (layers, dim, heads, patching), aur kya backbone **pretrained** hai (LoRA "fine-tune" kehta hai) — kis par?
2. LoRA rank, kin layers par.
3. Local epochs, learning rate, batch size, optimizer.
4. A(K) kaise nikala (final-round acc? kitne seeds?). "Scripts on request" — code public nahi.
5. FedProx/CL ki exact settings (µ ke ilawa).

## Paper ke Results jinhein Compare Karna Hai

- **Table 3:** K* aur accuracy har B par (K*: 2, 4, 5, 5, 12, 10; accuracy 85.1 → 96.6%, saturation ~96.6%).
- **Fig 2:** accuracy vs rounds — LoRaC-GA vs FedAvg vs FedProx vs CL (budget-matched).
- **Fig 3:** fitness vs iterations — GA vs greedy vs random.
- Claim: "up to 45% less comm at similar accuracy".
- **Out of scope:** P2P `log K` overlay (Section 3.4/4.1) — paper ke simulation results mein bhi evaluate nahi hua. TiFL/Oort/RBPS bars (Fig 5) bhi reproduce nahi karne.

## ⚠️ Paper mein Honest Red Flags (replication se pehle jaan lein)

1. **Budget kabhi bind nahi karta.** `R·S = 0.833 MB` per client. `B/C(K) ≥ 1` jab tak `K ≤ B/0.833` (B=10 → K≤12; B=50 → K≤60; B≥100 → har K≤100 feasible). Accuracy `A ≤ 1`, to `min(A, B/C) = A` is region mein → GA sirf `argmax A(K)` karta hai, budget ka koi asar nahi.
1b. **Aur sakht:** feasible set (`C ≤ B`) par `B/C ≥ 1 ≥ A` hamesha, to `min(A, B/C) = A` *har* feasible K par — budget sirf cap `K ≤ B/(R·S)` ke zariye kaam karta hai, min term kabhi bind nahi karta. GA effectively `argmax A(K)` over `K ≤ cap(B)` hai.
2. **Table 3 apne hi formula se match nahi karta.** B=10 par K=5 feasible hai (cost 4.17 MB) aur A=93.1% > K=2 ki 85.1% — to formula K=2 nahi chunta. B=100/400/1000/4000 par sab K feasible hain, phir bhi K* = 5, 5, 12, 10 (non-monotonic; 12 aur 10 dono 96.6%).
3. **96.6% shak-aana hai.** EMNIST-Balanced par centralized best published results ~91% ke aas-paas hain (meri yaad ke mutabiq — verify karein). Non-IID FL, R=10 mein 96.6% reproduce hona mushkil hai.
4. **S paper ke EMNIST model se nahi, FeDeRA (NLP) se liya gaya.**

**Is ka matlab:** exact numbers match karna target nahi. Target = **trend** (A(K) saturate hota hai? GA greedy/random se behtar? FedAvg/FedProx par comm saving?) + red flags ko thesis mein *documented observation* banana. Red flag #1 aap ke LLM setup ke liye bhi sabaq hai: M2 mein `S = 2.9583 MB`, K=10 → 591.7 MB, to B ∈ {10…400} wahan **asal mein bind karta hai** — ye aap ka M2+ setup paper se mazboot banata hai.

## Definition of Done

- [ ] EMNIST-Balanced loader + Dirichlet(α=0.3, 100 clients) split (`dirichlet.py` reuse)
- [ ] LoRA-wrapped compact transformer; `S` measured aur `0.0833 MB` se compare (rank/layers tune karke close lana)
- [ ] FedAvg loop EMNIST ke liye (aggregation reuse)
- [ ] A(K) profile K-grid par (Kaggle, manual run), seed 42
- [ ] GA (K-only) `src/ga/*` stubs mein implement + `pytest` pass; greedy + random baselines
- [ ] Table 3 analogue (K*, acc, cost per B) + Fig 3 analogue (GA vs greedy vs random)
- [ ] FedAvg / FedProx(µ=0.01) / CL budget-matched curve (Fig 2 analogue)
- [ ] **Paper-vs-ours table** + red-flag observations likhe hue
- [ ] Sir ko summary (Friday note)

## Kya Reuse Hota Hai / Kya Naya

Reuse: `src/fl/dirichlet.py`, `fedavg_aggregate`, `select_clients`, `src/utils/{checkpoint,metrics,plots}.py`.
Naya: `src/data/emnist_loader.py`, `src/models/emnist_lora.py`, EMNIST client/round loop (`client.py` HF-specific hai, to alag thin variant), `configs/m0_emnist.yaml`, Kaggle notebook, GA implementation (stubs bharna).
GA M4 mein `(K, r)` tak extend hoga — isliye K-only core saaf rakhein.

## Time Box

1 hafta (Kaggle: ~2–4 GPU hrs — chhota model, 28×28 images). Agar paper ka setup zyada samay le, to Fig 2 (FedProx/CL) sab se pehle cut karein; Table 3 + Fig 3 core hain.

## Analysis chalana (Kaggle ke baad, local, seconds)

```bash
git fetch && git checkout origin/kaggle-results-m0 -- results/m0_emnist
python -m src.ga.analysis --surface results/m0_emnist/A_Kr_surface.json --r 7 --out results/m0_emnist/analysis
```
Output: `analysis.json`, `comparison.md` (paper-vs-ours table + computed observations), `convergence.{png,pdf}` (Fig 3 analogue). Fig 2 analogue (FedAvg/FedProx/CL) alag step (3f).
