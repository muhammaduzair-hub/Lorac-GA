# LoRaC-GA on Real LLMs — Bandwidth-Aware Client Selection for Federated LoRA Fine-Tuning

A master's thesis project that extends **LoRaC-GA** (Solat & Lee, *Sensors* 2025) — a genetic-algorithm framework that selects the optimal number of participating clients per round under a bandwidth budget — from a proxy image benchmark (EMNIST) to **real LLM fine-tuning on a natural-language task**. Built on top of [FederatedScope-LLM](https://github.com/alibaba/FederatedScope/tree/llm).

> **Status:** 🚧 Work in progress (research prototype). Experiments are run in simulation on a single GPU.

---

## Research Question

> Does genetic-algorithm optimization of the per-round client count *K* under a fixed bandwidth budget *B* — originally validated on EMNIST images — transfer to LoRA fine-tuning of a small LLM on a GLUE text-classification task, while preserving communication savings under non-IID client data?

---

## Motivation

Federated learning lets many clients collaboratively fine-tune a model without sharing raw data, but transmitting large LLM updates each round is expensive on bandwidth-limited edge links. LoRA reduces update size by freezing the backbone and training only small adapters. A remaining question is *how many* clients should participate per round to balance accuracy against communication cost under a budget. The base paper (LoRaC-GA) answers this with a genetic algorithm — but validates only on image data despite framing the work around LLMs. This project closes that gap by testing the method where it actually matters: real language models on real language tasks.

---

## Approach in One Line

Freeze a small LLM → inject LoRA adapters → profile accuracy *A(K)* for candidate client counts → run a genetic algorithm to maximize `f(K) = min(A(K), B / (R·K·S))` under the budget → evaluate the selected *K\** against federated baselines.

---

## Key Variables

| Symbol | Meaning | Default |
|--------|---------|---------|
| `K` | Clients selected per round (decision variable) | 1 … K_max |
| `K_max` | Total clients | 100 |
| `R` | Communication rounds | 10 |
| `S` | LoRA adapter size per client (MB) | model-dependent |
| `B` | Bandwidth budget (MB) | {10, 50, 100, 400, 1000, 4000} |
| `C(K)` | Total communication cost = `R·K·S` | — |
| `A(K)` | Empirical accuracy with K clients | profiled offline |
| `f(K)` | Fitness = `min(A(K), B/C(K))` | — |

---

## Repository Layout

```
thesis_LoRaC_GA/
├── CLAUDE.md                 # AI-assistant project context (internal)
├── README.md                 # this file
├── requirements.txt          # local (CPU) deps for code + unit tests
├── requirements-kaggle.txt   # Kaggle (GPU) deps for training
├── configs/                  # YAML configs (LoRA, FL setup, GA search)
├── src/
│   ├── ga/                   # genetic algorithm: fitness, operators, search
│   ├── fl/                   # bridge to FederatedScope client selection
│   ├── data/                 # GLUE loaders + non-IID Dirichlet split
│   ├── models/               # LoRA adapter wrapper
│   └── utils/                # checkpointing, metrics
├── notebooks/                # Kaggle bootstrap + experiment notebooks
├── results/                  # profiles, GA logs, plots (small files only)
├── tests/                    # pytest unit tests (run without GPU)
└── third_party/
    └── FederatedScope/       # base framework (git submodule, llm branch)
```

---

## Workflow

This project uses a **local-authoring / Kaggle-training** split:

- **Local machine** — write code, run unit tests (no GPU needed), commit, push to GitHub.
- **Kaggle Notebooks** — clone the repo, install FederatedScope, run GPU training, push results back.

```
LOCAL (write + test + commit)  ──push──►  GitHub  ──clone──►  KAGGLE (install + train)
                                             ▲                        │
                                             └────── results push ────┘
```

FederatedScope is intentionally **not installed locally** (heavy dependencies); it is installed only inside the Kaggle notebook.

---

## Setup

### 1. Clone (with submodule)

```bash
git clone --recurse-submodules https://github.com/USERNAME/thesis-lorac-ga.git
cd thesis-lorac-ga
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Local environment (code + tests)

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run unit tests (no GPU required)

```bash
python -m pytest tests/ -v
```

---

## Running Training (Kaggle)

1. Create a new Kaggle Notebook.
2. Enable **GPU (T4)** and **Internet** in the notebook settings.
3. Run the bootstrap notebook `notebooks/00_kaggle_bootstrap.ipynb`, which:
   - clones this repo with submodules,
   - installs `requirements-kaggle.txt`,
   - installs FederatedScope via `pip install -e third_party/FederatedScope`,
   - runs the unit tests as a sanity check,
   - verifies the GPU is available.
4. Run the experiment notebooks to profile *A(K)*, execute the GA search, and evaluate.

Results are written to `results/` and pushed back to a dedicated `kaggle-results-*` branch.

---

## Method Details

- **Model:** small LLM backbone (e.g., DistilBERT-base / RoBERTa-base) with LoRA adapters (rank `r = 8`, `α = 16`).
- **Task:** GLUE text classification (e.g., SST-2 / MNLI).
- **Data split:** non-IID Dirichlet partition (`α = 0.3`) across `K_max = 100` simulated clients.
- **Optimizer over K:** genetic algorithm (population `P = 20`, generations `G = 30`, crossover `p_c = 0.5`, mutation `p_m = 0.2`, elitism ratio `0.1`).
- **Objective:** max-min `f(K) = min(A(K), B/(R·K·S))` subject to `1 ≤ K ≤ K_max` and `C(K) ≤ B`.

---

## Scope

**In scope:** one small LLM, one GLUE task, GA-based *K* selection under a budget, non-IID data, simulation on a single GPU, comparison against two baselines (FedAvg/FedIT and a recent federated-LoRA method).

**Out of scope (future work):** billion-parameter models, generation/instruction tuning, differential privacy, secure aggregation, heterogeneous client resources / mixed LoRA ranks, and hardware testbeds.

---

## Reproducibility

- All randomness is seeded (`seed = 42`) across `torch`, `numpy`, and `random`.
- All hyperparameters live in `configs/` — no hard-coded values.
- Each experiment run records its config and git commit hash.

---

## Acknowledgements & Attribution

This project builds directly on:

- **Base method:** F. Solat and J. Lee, "Optimizing Client Participation in Communication-Constrained Federated LLM Adaptation with LoRA," *Sensors*, 25(21):6538, 2025. https://doi.org/10.3390/s25216538
- **Base framework:** W. Kuang et al., "FederatedScope-LLM: A Comprehensive Package for Fine-tuning LLMs in Federated Learning," *KDD* 2024. https://arxiv.org/abs/2309.00363
- **LoRA:** E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR* 2022.

Development was assisted by AI coding tools (Claude Code); all research design, experiments, and analysis are the author's own.

---

## License

- This project's original code: MIT (see `LICENSE`).
- `third_party/FederatedScope` retains its own **Apache-2.0** license.

---

## Author

**Muhammad Uzair** (2500514) — Creative technology, Air University
Supervisor: **Dr Adnan Aslam**

---

## Selected References

1. F. Solat, J. Lee. *Optimizing Client Participation in Communication-Constrained Federated LLM Adaptation with LoRA.* Sensors, 2025.
2. B. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017.
3. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
4. W. Kuang et al. *FederatedScope-LLM.* KDD, 2024.
5. J. Bai et al. *Federated Fine-tuning of LLMs under Heterogeneous Tasks and Client Resources (FlexLoRA).* NeurIPS, 2024.
6. P. Guo et al. *Selective Aggregation for Low-Rank Adaptation in Federated Learning (FedSA-LoRA).* ICLR, 2025.
