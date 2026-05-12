"""
Central configuration for the CAI pipeline.
All hyperparameters in one place — change values here OR override via
Streamlit dashboard sliders (streamlit_dashboard.py passes config_overrides dict).

Pipeline overview:
  Phase 1: SFT on critique+revision pairs from HH-RLHF + Groq API → π_ref
  Phase 2: GRPO RL with constitutional reward + KL penalty            → π_θ

Paper: Bai et al. 2022 — Constitutional AI: Harmlessness from AI Feedback
"""

import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on shell env vars

# =============================================================================
# Model
# =============================================================================
# Options:
# - "Qwen/Qwen2-0.5B-Instruct"  # Fast, ~2-3GB VRAM, good for testing
# - "Qwen/Qwen2-1.5B-Instruct"  # Better quality, ~4-5GB VRAM, recommended
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"  # Using 1.5B for better quality
MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True              # QLoRA — 4-bit quantized base, FP16 adapters

# =============================================================================
# LoRA Adapter
# =============================================================================
LORA_RANK = 16                   # Adapter bottleneck rank
LORA_ALPHA = 16                  # Scaling factor (effective LR scale = alpha/rank = 1)
LORA_DROPOUT = 0.0               # Unsloth recommends 0 for stability
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
    "gate_proj", "up_proj", "down_proj",        # MLP / FFN
]

# =============================================================================
# SFT Training (Phase 1) — Paper Section 3
# =============================================================================
SFT_LEARNING_RATE = 2e-4
SFT_BATCH_SIZE = 4
SFT_GRAD_ACCUM = 4               # Effective batch = 4 * 4 = 16
SFT_EPOCHS = 3
SFT_WARMUP_STEPS = 10
SFT_OPTIMIZER = "adamw_8bit"     # Paged AdamW in 8-bit
SFT_MAX_SAMPLES = 2000           # Subset of HH-RLHF for Colab T4 limits

# =============================================================================
# GRPO Training — KL-Regularized (Phase 2) — Paper Section 4
# =============================================================================
GRPO_NUM_GENERATIONS = 4         # G — responses sampled per prompt per step
GRPO_MAX_NEW_TOKENS = 256        # Max response length during rollout
GRPO_LEARNING_RATE = 5e-6        # Very small — RL is unstable with large LR
GRPO_BATCH_SIZE = 2              # Prompts per device per step
GRPO_GRAD_ACCUM = 4
GRPO_EPOCHS = 1
GRPO_KL_COEFF = 0.1              # β — KL penalty: R_KL = score - β*KL(π_θ||π_ref)
                                  # TRL 0.24+ param name: beta= in GRPOConfig
GRPO_MAX_PROMPTS = 500           # Prompts in GRPO training loop

# =============================================================================
# Groq API — Judge + Critic (replaces paper's pre-trained LM feedback model)
# Paper Section 4.1: "present response pair to feedback model with a principle"
# =============================================================================
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
                                  # Default: 8B model — higher free-tier token limits
                                  # Override in .env: GROQ_MODEL=llama-3.3-70b-versatile

# Support multiple keys for rotation to avoid rate limits
GROQ_API_KEYS = []
for i in range(1, 6):
    _key = os.environ.get(f"GROQ_API_KEY_{i}")
    if _key:
        GROQ_API_KEYS.append(_key)

# Fallback to GROQ_API_KEY if none of the above are set
_single_key = os.environ.get("GROQ_API_KEY", "")
if _single_key and _single_key not in GROQ_API_KEYS:
    GROQ_API_KEYS.append(_single_key)

_key_index = 0

def get_next_groq_api_key():
    """Returns the next Groq API key in the rotation."""
    global _key_index
    if not GROQ_API_KEYS:
        return ""
    key = GROQ_API_KEYS[_key_index % len(GROQ_API_KEYS)]
    _key_index += 1
    return key

GROQ_API_KEY = get_next_groq_api_key()
GROQ_TEMPERATURE = 0.0           # Deterministic judging for reproducibility
API_DELAY_SECONDS = 0.5          # Groq is fast (~300 tok/s), short delay OK
API_MAX_RETRIES = 10             # Max retries on 429/5xx before giving up
API_RETRY_BASE_DELAY = 5         # Base delay (seconds) for exponential backoff

# =============================================================================
# Reward Scoring — Paper Section 4.3
# =============================================================================
REWARD_SAFE = 1.0                # Fully compliant response
REWARD_PARTIAL = 0.5             # Partially compliant
REWARD_HARMFUL = 0.0             # Non-compliant
# Soft label clamping (paper Section 4.3: "clamping at 40-60 improved results")
REWARD_CLAMP_LOW = 0.4
REWARD_CLAMP_HIGH = 0.6
# Number of principles to ensemble per reward call (paper Section 4.3)
REWARD_ENSEMBLE_N = 4

# =============================================================================
# Checkpoint / Resume — Critical for Colab free tier disconnects
# =============================================================================
CHECKPOINT_DIR = "./outputs/checkpoints"
CHECKPOINT_EVERY_N_STEPS = 50   # Save every N optimizer steps
RESUME_FROM_CHECKPOINT = True   # Auto-detect and resume if checkpoint exists

# =============================================================================
# File Paths
# =============================================================================
DATASET_DIR = "./datasets"
SFT_DATA_PATH = "./datasets/sft_dataset.jsonl"
GRPO_PROMPT_PATH = "./datasets/grpo_prompts.jsonl"
SFT_OUTPUT_DIR = "./outputs/sft_model"
GRPO_OUTPUT_DIR = "./outputs/grpo_model"
LOG_DIR = "./logs"
REWARD_LOG_PATH = "./logs/reward_log.jsonl"
SFT_LOG_PATH = "./logs/sft_log.jsonl"
GRPO_LOG_PATH = "./logs/grpo_log.jsonl"

# =============================================================================
# TensorBoard
# =============================================================================
TENSORBOARD_LOG_DIR = "./logs/tensorboard"
