# Constitutional AI — Complete Training System
## Implementing Bai et al. 2022 with GRPO-KL on Qwen2-0.5B

> **KL-Regularized GRPO** replaces the paper's PPO with group-relative reward normalization  
> and a KL penalty term: **R_KL(s,a) = constitutional_score(s,a) − β · KL(π_θ ∥ π_ref)**

---

### Quick Start — Google Colab

```bash
# 1. Open colab_notebook.ipynb in Colab
# 2. Cell 6: set GROQ_API_KEY (free at console.groq.com)
# 3. Run all cells in order
# 4. Checkpoints auto-save to Google Drive every 50 steps
```

### Quick Start — Local Dashboard

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
streamlit run streamlit_dashboard.py
```

---

### Architecture

| Component | Choice | Paper Equivalent |
|---|---|---|
| Base model | Qwen2-0.5B-Instruct (4-bit QLoRA) | 52B helpful RLHF model |
| SFT framework | Unsloth + TRL SFTTrainer | SL-CAI supervised stage |
| RL framework | TRL GRPOTrainer (KL-regularized) | PPO with KL penalty |
| Feedback model | llama-3.3-70b via Groq API | Pre-trained LM feedback model |
| Constitution | 16 principles from Appendix C | Paper verbatim |
| Dataset | Anthropic/hh-rlhf (HuggingFace) | Same dataset as paper |

### RL Formulation

| Term | Implementation |
|---|---|
| State s | User prompt |
| Action a | Full generated response |
| R(s,a) | Groq constitutional score (ensembled, 4 principles) |
| R_KL(s,a) | R(s,a) − β · KL(π_θ ∥ π_ref) |
| β | 0.1 (KL penalty coefficient) |
| Algorithm | GRPO, G=4 group normalization |

### Paper Fidelity

| Paper Component | Implementation |
|---|---|
| 16 RL principles (Appendix C.1) | `constitution.json` — verbatim |
| Critique+revision chain (Sec 3.1) | `src/data/prepare_datasets.py` |
| Principle ensembling (Sec 4.3) | `reward_function.py::compute_ensembled_reward()` |
| Soft labels clamped to [0.4,0.6] (Sec 4.3) | `get_groq_constitutional_score()` |
| Evasiveness penalty (Sec 4.4) | `detect_evasive_response()` |
| Absolute harmfulness 0-4 scale (Sec 4.5) | `evaluator.py` |
| Elo evaluation (Sec 3.3) | `compute_pairwise_elo()` via Groq |

### Project Structure

```
constitutional-ai/
├── colab_notebook.ipynb          # 15-cell Colab training notebook
├── streamlit_dashboard.py        # 9-page training/inference dashboard
├── constitution.json             # 16 SL + 16 RL principles (paper Appendix C)
├── config.py                     # All hyperparameters
├── graph.html                    # Interactive codebase visualization
├── src/
│   ├── data/
│   │   ├── download_datasets.py  # HF HH-RLHF download (parquet + rows API)
│   │   └── prepare_datasets.py   # Critique+revision SFT data; GRPO prompts
│   ├── training/
│   │   ├── phase1_sft.py         # SFT training loop → π_ref
│   │   ├── phase2_grpo.py        # KL-GRPO training loop → π_θ
│   │   ├── reward_function.py    # Constitutional reward via Groq API
│   │   └── checkpoint_manager.py # Save/resume + Google Drive sync
│   ├── evaluation/
│   │   └── evaluator.py          # Pairwise Elo + harmfulness metrics
│   └── visualization/
│       ├── tensorboard_logger.py # TensorBoard SummaryWriter wrapper
│       └── graph_generator.py    # Generates graph.html (pyvis)
├── datasets/                     # Downloaded HH-RLHF JSONL files
├── outputs/                      # Model checkpoints
│   ├── sft_model_merged/         # π_ref (merged 16-bit)
│   ├── grpo_model_merged/        # π_θ (merged 16-bit)
│   ├── grpo_model_gguf/          # Q4_K_M GGUF for local inference
│   └── checkpoints/              # Per-step emergency checkpoints
└── logs/                         # TensorBoard events + JSON logs
    ├── tensorboard/
    ├── reward_log.jsonl
    ├── sft_log.jsonl
    └── grpo_log.jsonl
```

### TRL 0.24.0 API Notes

> **IMPORTANT** — TRL 0.24.0 silently renamed several parameters:
>
> | Old (pre-0.24) | New (0.24+) | Effect if wrong |
> |---|---|---|
> | `kl_coeff=` in GRPOConfig | `beta=` | KL penalty silently disabled |
> | `max_new_tokens=` | `max_completion_length=` | Uses default length |
> | `tokenizer=` in Trainer | `processing_class=` | Silently ignored |
>
> This codebase uses the correct TRL 0.24.0 parameter names throughout.

### Checkpoint / Resume

Training saves every 50 steps and at each epoch end.  
On Colab reconnect, simply re-run the training cell:
```
[ckpt] Resuming from epoch 2, step 150 (loss: 0.892)
```
Checkpoints also sync to `/content/drive/MyDrive/cai_checkpoints/` if Drive is mounted.

### Estimated Costs

| Phase | Time (T4) | Groq API calls | Est. cost |
|---|---|---|---|
| Dataset download | ~5 min | 0 | $0 |
| Prepare SFT (2000 samples) | ~90 min | ~4000 | ~$0.20 |
| SFT training | ~45 min | 0 | $0 |
| GRPO training (500 prompts) | ~2 hrs | ~2000 | ~$0.10 |
| Evaluation | ~30 min | ~150 | ~$0.01 |

*Groq free tier: 14,400 req/day on llama-3.3-70b-versatile.*
