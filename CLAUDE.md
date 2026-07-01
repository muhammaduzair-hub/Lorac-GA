# CLAUDE.md — LoRaC-GA Thesis Project

> **Note for Claude Code**: Ye file aap ke project ka master context hai. Har session ke shuru mein read karein. Project root mein rakhi jaye, naam exactly `CLAUDE.md` ho.

---

## 1. Project ka Maqsad (Quick Summary)

Master's thesis project: **Bandwidth-aware client selection for federated LoRA fine-tuning of LLMs**.

Asal paper (Solat & Lee, *Sensors* 2025, "LoRaC-GA") ne EMNIST (handwritten images) par ek genetic algorithm propose kiya jo per-round mein optimal client count `K` decide karta hai bandwidth budget `B` ke andar. Lekin paper ne ye sirf images par test kiya — LLMs par nahi, halankay daawa LLM ka tha.

**Mera contribution**: Wahi GA-based K-selection ek **real LLM** par ek **real NLP task** par test karna, simulation mein, minimal GPU pe.

---

## 2. Research Question (sirf ek)

> Does GA-based optimization of per-round client count `K` under a fixed bandwidth budget `B` — originally validated on EMNIST — transfer to LoRA fine-tuning of a small LLM on a GLUE text classification task while preserving the communication savings under non-IID client data?

Single hypothesis, single pass/fail outcome.

---

## 3. In Scope vs Out of Scope

### In Scope ✅
- Ek chhota LLM: DistilBERT-base (66M params), ya RoBERTa-base (125M), ya GPT-2 small (124M)
- Ek task family: GLUE text classification (SST-2 ya MNLI)
- GA-based K-selection under bandwidth budget B
- Non-IID Dirichlet split (α = 0.3) across 100 simulated clients
- LoRA adapters only (rank r = 8, alpha = 16)
- 2 baselines: FedAvg/FedIT aur FedSA-LoRA
- Simulation only, single Colab/Kaggle T4 GPU

### Out of Scope ❌
- Billion-parameter LLaMA-class models
- Generation ya instruction-tuning
- Differential privacy / secure aggregation (future work)
- Heterogeneous client resources ya mixed LoRA ranks (future work)
- Multi-solver comparison (Bayesian opt, RL) — future work
- Hardware testbed (future work)

---

## 4. Tech Stack

| Component | Version / Choice |
|-----------|------------------|
| Base framework | FederatedScope-LLM (Alibaba, `llm` branch) |
| LLM library | HuggingFace transformers `4.36.0` |
| PEFT library | HuggingFace `peft 0.7.0` |
| FL backbone | PyTorch `2.x` |
| Dataset library | `datasets` (HuggingFace) |
| Compute | Google Colab Free (T4 16GB) + Kaggle Notebooks backup |
| Storage | Google Drive (checkpoints) |
| Language | Python 3.10+ |

---

## 5. Folder Structure

```
thesis_LoRaC_GA/
├── CLAUDE.md                    # ye file
├── README.md                    # public-facing readme
├── requirements.txt             # pinned dependencies
├── configs/
│   ├── base_lora.yaml          # LoRA + model config
│   ├── fl_setup.yaml           # federated setup
│   └── ga_search.yaml          # GA hyperparams
├── src/
│   ├── __init__.py
│   ├── ga/
│   │   ├── __init__.py
│   │   ├── chromosome.py       # K encoding
│   │   ├── fitness.py          # f(K) = min(A(K), B/C(K))
│   │   ├── operators.py        # selection, crossover, mutation
│   │   └── search.py           # main GA loop
│   ├── fl/
│   │   ├── server.py           # aggregation logic
│   │   ├── client.py           # local LoRA training
│   │   └── dirichlet.py        # non-IID split
│   ├── data/
│   │   └── glue_loader.py      # SST-2 / MNLI loaders
│   ├── models/
│   │   └── lora_wrap.py        # LoRA adapter wrapper
│   └── utils/
│       ├── checkpoint.py       # save/resume
│       └── metrics.py          # accuracy + comm cost
├── notebooks/
│   ├── 01_setup_test.ipynb     # Colab setup verification
│   ├── 02_profile_AK.ipynb     # accuracy vs K profiling
│   ├── 03_ga_search.ipynb      # main GA experiments
│   └── 04_baselines.ipynb      # FedAvg comparison
├── results/
│   ├── profiles/               # A(K) profiles
│   ├── ga_runs/                # GA search logs
│   └── final/                  # plots, tables
└── tests/
    └── test_ga.py              # unit tests for GA
```

---

## 6. Important Variables aur Symbols

| Symbol | Meaning | Default Value |
|--------|---------|---------------|
| `K` | Per-round selected client count | Decision variable (1 to 100) |
| `K_max` | Total available clients | 100 |
| `R` | Total communication rounds | 10 |
| `S` | LoRA adapter size per client (MB) | Depends on model + rank |
| `B` | Bandwidth budget (MB) | {10, 50, 100, 400, 1000, 4000} |
| `C(K)` | Total comm cost | `R × K × S` |
| `A(K)` | Empirical accuracy with K clients | Profiled offline |
| `f(K)` | Fitness function | `min(A(K), B/C(K))` |
| `α` | Dirichlet concentration | 0.3 (non-IID) |
| `r` | LoRA rank | 8 |
| `P` | GA population size | 20 |
| `G` | GA generations | 30 |
| `p_c` | Crossover probability | 0.5 |
| `p_m` | Mutation probability | 0.2 |

---

## 7. Coding Conventions

- **Python style**: PEP 8, max line 100 chars
- **Type hints**: Required for all public functions
- **Docstrings**: Google style, with `Args`, `Returns`, `Raises`
- **Logging**: Use `logging` module, NOT `print` (except notebooks)
- **Random seeds**: Always set — `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` — taa ke results reproducible hon
- **Config**: YAML files, loaded via `omegaconf` ya `hydra` (no hard-coded params)
- **Tests**: Pytest for GA logic; unit tests must pass before merging
- **Commits**: Atomic, descriptive messages (e.g. `feat(ga): add tournament selection`)

---

## 8. Reproducibility Rules (Sabse Important)

1. **Har experiment ka seed log karein** — config file mein `seed: 42` always
2. **Checkpoints har round ke baad save karein** — Google Drive par
3. **Run identifier**: Har experiment ka unique ID (timestamp + git commit hash)
4. **No GPU-specific tricks** — code CPU-fallback honi chahiye debugging ke liye
5. **Pinned dependencies** — `requirements.txt` mein exact versions

---

## 9. Aap (Claude Code) Se Kya Kaam Karwana Hai

### Aksar wale tasks:
- **Code likhna**: GA operators (selection, crossover, mutation), fitness function, Dirichlet splitter, LoRA wrapper
- **Code review**: Mera likha hua code check karna — bugs, style, efficiency
- **Refactoring**: Lambi functions ko chhote, modular pieces mein todna
- **Debug**: Errors solve karne mein madad
- **Tests likhna**: Pytest tests for new functions
- **Docstrings add karna**: Existing functions mein documentation add karna
- **Plot banana**: Matplotlib/seaborn se publication-quality plots
- **Latex tables**: Results ko LaTeX tabular format mein convert karna

### Aap (Claude Code) ko kya **nahi** karna:
- **Actual training run nahi karwana** — woh manually Colab/Kaggle par hoga
- **Hyperparameters guess nahi karne** — agar unclear ho to puchein
- **Large refactors bina poochay nahi** — pehle plan dikhayen
- **External services call nahi karne** — sirf local code edit
- **Random research papers ki advice nahi** — sirf is project ka context

---

## 10. Common Commands

```bash
# Environment setup (Colab par)
pip install -r requirements.txt

# Local GA test (no GPU needed)
python -m src.ga.search --config configs/ga_search.yaml --dry-run

# Profile A(K) for a single K value
python -m src.fl.server --config configs/fl_setup.yaml --K 5

# Run full GA search (needs profile data)
python -m src.ga.search --config configs/ga_search.yaml --budget 100

# Baseline comparison
python -m src.fl.server --baseline fedavg --K 10 --budget 100

# Run tests
pytest tests/ -v
```

---

## 11. Known Issues / Watchlist

- **Colab disconnect**: Free tier 8-hour limit. Always checkpoint to Drive.
- **OOM on T4**: If RoBERTa-base + batch 16 OOMs, drop to batch 8 + grad accumulation 2.
- **Dirichlet edge case**: With α=0.3 and 100 clients, some clients can get 0 samples for a class. Filter or resample.
- **GA stuck in local optimum**: If mutation rate too low, fitness plateaus early. Tune `p_m` between 0.15–0.25.
- **FederatedScope version pin**: Use `llm` branch, NOT `master`. Master doesn't have LLM modules.

---

## 12. Reference Papers (jo zaroor padhe rakhne hain)

1. **Solat & Lee, 2025** — Original LoRaC-GA paper (*Sensors*, DOI: 10.3390/s25216538)
2. **Kuang et al., 2024** — FederatedScope-LLM (KDD 2024)
3. **Hu et al., 2022** — Original LoRA paper (ICLR 2022)
4. **Bai et al., 2024** — FlexLoRA (NeurIPS 2024)
5. **Guo et al., 2025** — FedSA-LoRA (ICLR 2025)

---

## 13. Sir (Supervisor) ke Liye Updates

- **Frequency**: Har Friday ek brief progress note (3-5 bullet points)
- **Format**: "Ye hua, ye stuck hai, agle hafte ye karoonga"
- **Major decisions**: Sir se confirm karne ke baad commit
- **Conflicts**: Agar baseline result expected se bohot alag aaye, sir ko tunant batayen

---

## 14. Aakhri Reminder for Claude Code

- **Scope se bahar mat jaye** — agar koi feature is project ka hissa nahi (privacy, hardware, billion-param models), to remind karein ke ye out of scope hai
- **Reproducibility hamesha pehli priority** — seed, config, checkpoint ka khayal
- **Honest feedback** — agar mera approach galat lage to seedha batain, taarif nahi
- **Roman Urdu mein code comments theek hain** — agar aap chahein, lekin docstrings English mein rakhein (international code conventions)
- **Pakistan-specific context**: Author Pakistani master's student hai, supervisor "Sir" kehta hai — culturally respectful tone use karein
