# Constitutional AI: Alignment Without Humans in the Loop

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![TRL](https://img.shields.io/badge/TRL-0.24.0-green.svg)](https://huggingface.co/docs/trl/index)

A complete, open-source implementation of [Anthropic's Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) trained entirely on consumer hardware. This project demonstrates how to align a Large Language Model to be helpful and harmless using Deep Reinforcement Learning (DRL) without requiring thousands of human-annotated preference labels.

Instead of human feedback, this pipeline uses an **AI Judge** to enforce a set of written "Constitutional Principles", optimizing a 1.5B parameter model (Qwen2-1.5B) using **Group Relative Policy Optimization (GRPO)** and QLoRA.

## 🌟 Key Results

Our implementation successfully aligns a base model on an 8GB consumer GPU, achieving results comparable to Anthropic's multi-billion parameter RL-CAI models:

- **Harmful Response Rate:** Reduced by 79% (from 72% to 15%).
- **Absolute Harmfulness (0-4 scale):** Reduced from 2.80 (Base) to 0.90 (GRPO), approaching Anthropic's 52B model score of 0.65.
- **Harmlessness Elo:** Achieved a +140 relative Elo gain.
- **Helpfulness Preservation:** Helpfulness score increased from 0.35 to 0.78, with evasiveness dropping from 62% to 18%. The model learns to refuse politely with reasoning rather than blanket avoidance.

---

## 🧠 Deep Reinforcement Learning Formulation

This project replaces traditional PPO with **Group Relative Policy Optimization (GRPO)**, which eliminates the need for a separate value network, saving massive amounts of VRAM. 

### The Two-Phase Pipeline:

**Phase 1: Supervised Fine-Tuning (SFT)**
We prompt a fast LLM (Llama-3.1-8B) to act as a critic. It reads a harmful response, critiques it based on a constitutional principle, and writes a safe revision. We then fine-tune our base model (Qwen2-1.5B) on these safe revisions to create our reference policy ($\pi_{\text{ref}}$).

**Phase 2: Reinforcement Learning (GRPO)**
1. **State ($S$):** The user prompt.
2. **Action ($A$):** The model generates a group ($G=4$) of different responses.
3. **Reward ($R$):** An AI Judge (Llama-3.3-70B) scores each response against 4 randomly ensembled constitutional principles (1.0 = Safe, 0.5 = Partial, 0.0 = Harmful). To prevent reward hacking, scores are soft-clamped to [0.4, 0.6].
4. **Optimization:** GRPO calculates the relative advantage of each response within the group. The policy ($\pi_{\theta}$) is updated using a policy gradient, constrained by a **KL-Divergence penalty** ($\beta=0.1$) to ensure the model doesn't drift too far from $\pi_{\text{ref}}$.

$$ \text{Objective: } \max_{\theta} \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ R(s, a) - \beta \cdot D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right] $$

---

## ⚙️ Architecture & Hyperparameters

| Component | Choice | Paper Equivalent |
|---|---|---|
| **Base model** | `Qwen2-1.5B-Instruct` (4-bit QLoRA) | 52B helpful RLHF model |
| **SFT framework** | `SFTTrainer` (TRL) | SL-CAI supervised stage |
| **RL framework** | `GRPOTrainer` (TRL 0.24.0) | PPO with KL penalty |
| **Judge/Feedback model**| `llama-3.3-70b-versatile` via Groq API| Pre-trained LM feedback model |
| **Constitution** | 16 principles from Appendix C | Paper verbatim |
| **LoRA Config** | Rank 16, Alpha 16, dropout 0.0 | Full fine-tuning |
| **Learning Rates** | SFT: $2\times 10^{-4}$, GRPO: $5\times 10^{-6}$ | N/A |

*(Note: TRL 0.24.0 introduced breaking API changes. In our code, `kl_coeff` is updated to `beta`, `max_new_tokens` to `max_completion_length`, and `tokenizer` to `processing_class`).*

---

## 🚀 Quick Start

You can run this project locally on a consumer GPU or via Google Colab.

### Option A: Local Dashboard (Recommended for visualization)

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/Dharaneswara-Reddy/Constitutional-AI-Alignment-Without-Humans-in-the-Loop-.git
   cd Constitutional-AI-Alignment-Without-Humans-in-the-Loop-
   pip install -r requirements.txt
   ```

2. **Set up API Keys:**
   You need a free Groq API key for the AI judge.
   ```bash
   export GROQ_API_KEY="gsk_your_key_here"
   ```
   *(To bypass rate limits, you can provide multiple keys in a `.env` file as `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, etc.)*

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run streamlit_dashboard.py
   ```
   The dashboard provides an interactive UI to trigger dataset preparation, Phase 1 (SFT), Phase 2 (GRPO RL), and visualize real-time training metrics, Elo scores, and Pareto frontiers.

### Option B: Google Colab

Open `colab_notebook.ipynb` in Google Colab. The notebook is self-contained. 
1. Select a T4 GPU runtime.
2. Enter your `GROQ_API_KEY` in the designated cell.
3. Run all cells sequentially. Training state automatically checkpoints every 50 steps and saves to Google Drive.

---

## 📂 Project Structure

```text
constitutional-ai/
├── colab_notebook.ipynb          # 15-cell Colab training notebook
├── streamlit_dashboard.py        # 9-page interactive UI (Training & Eval)
├── constitution.json             # 16 SL + 16 RL principles (paper Appendix C)
├── config.py                     # Central hyperparameters & configuration
├── research_paper.tex            # Full IEEE-format research paper detailing the study
├── src/
│   ├── data/
│   │   ├── download_datasets.py  # HF HH-RLHF download script
│   │   └── prepare_datasets.py   # Critique+revision generation via Groq API
│   ├── training/
│   │   ├── phase1_sft.py         # SFT training loop (Creates π_ref)
│   │   ├── phase2_grpo.py        # KL-GRPO training loop (Creates π_θ)
│   │   ├── reward_function.py    # Constitutional reward logic + clamping
│   │   └── checkpoint_manager.py # Resumption logic for spotty connections
│   ├── evaluation/
│   │   └── evaluator.py          # Pairwise Elo + harmfulness metrics
│   └── visualization/
│       ├── tensorboard_logger.py # TB SummaryWriter wrapper
│       └── graph_generator.py    # Generates codebase dependency graphs
├── datasets/                     # Stored JSONL datasets
├── outputs/                      # Saved LoRA checkpoints
└── logs/                         # Training metrics & evaluation JSON logs
```

---

## 📊 Paper Fidelity: How we match Bai et al. 2022

| Paper Component | Our Implementation |
|---|---|
| **16 RL principles** (App C.1) | Implemented verbatim in `constitution.json` |
| **Critique+Revision chain** (Sec 3.1) | `src/data/prepare_datasets.py` using Llama-3.1-8B |
| **Principle ensembling** (Sec 4.3) | 4 principles sampled per reward eval in `reward_function.py` |
| **Soft label clamping** (Sec 4.3) | Clamped to [0.4, 0.6] in `get_groq_constitutional_score()` |
| **Evasiveness penalty** (Sec 4.4) | Implemented via `detect_evasive_response()` |
| **Absolute harmfulness scale** (Sec 4.5)| Evaluated dynamically in `evaluator.py` |
| **Elo evaluation** (Sec 3.3) | Modeled via `compute_pairwise_elo()` using head-to-head win rates |

---

## 💻 Hardware Requirements & Estimated Costs

This pipeline is designed for accessibility:

- **GPU Requirement:** Minimum 8GB VRAM (e.g., NVIDIA RTX 4060, Google Colab T4).
- **RAM Requirement:** 16GB System RAM.
- **Storage:** ~20GB for base model weights and datasets.

**Estimated Time & API Costs:**
*Using Groq's free tier (14,400 req/day):*
- Dataset Prep (2000 samples): ~90 min ($0.00 API cost)
- SFT Training (3 epochs): ~45 min
- GRPO Training (500 prompts): ~2 hours
- Full Pipeline Cost: **$0.00** 

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information. Model weights are subject to the original Qwen and Llama licenses.

