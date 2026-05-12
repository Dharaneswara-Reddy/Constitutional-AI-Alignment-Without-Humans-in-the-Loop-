"""
Phase 2: KL-Regularized GRPO — Constitutional AI RL Phase.
Paper: Bai et al. 2022, Section 4.
"""

import json, os, sys, torch, time, threading
from pathlib import Path
from collections import OrderedDict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- MONKEY PATCH FOR TRANSFORMERS 5.5.0 + PYTORCH < 2.5 ---
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        if target == "":
            raise ValueError("Cannot set an empty string as a submodule name.")
        atoms = target.split(".")
        name = atoms[-1]
        parent = self.get_submodule(".".join(atoms[:-1])) if len(atoms) > 1 else self
        setattr(parent, name, module)
    torch.nn.Module.set_submodule = _set_submodule
# -----------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (SFT_OUTPUT_DIR, GRPO_OUTPUT_DIR, GRPO_PROMPT_PATH,
                    MAX_SEQ_LENGTH, LOAD_IN_4BIT, BASE_MODEL,
                    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
                    GRPO_NUM_GENERATIONS, GRPO_MAX_NEW_TOKENS, GRPO_LEARNING_RATE,
                    GRPO_BATCH_SIZE, GRPO_GRAD_ACCUM, GRPO_EPOCHS,
                    GRPO_KL_COEFF,
                    GRPO_MAX_PROMPTS, TENSORBOARD_LOG_DIR, GRPO_LOG_PATH, LOG_DIR,
                    get_next_groq_api_key, API_DELAY_SECONDS, API_MAX_RETRIES, API_RETRY_BASE_DELAY)
from src.training.checkpoint_manager import CheckpointManager
from src.visualization.tensorboard_logger import TensorBoardLogger
from src.data.prepare_datasets import groq_call_with_retry

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import get_peft_model, LoraConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset


# =========================================================================
# Rate-limited reward scorer
# =========================================================================

# LRU cache — avoids re-scoring identical (prompt, completion) pairs
_score_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 2000
_api_lock = threading.Semaphore(1)   # Only 1 Groq call at a time


def _get_groq_client():
    """Returns a Groq client. Add extra keys here for rotation."""
    from groq import Groq
    key = get_next_groq_api_key()
    if not key:
        raise ValueError("GROQ_API_KEY not set.")
    return Groq(api_key=key)


def _score_single(prompt: str, completion: str) -> float:
    """
    Score one (prompt, completion) pair via Groq.
    Uses LRU cache + serialised API access + exponential backoff.
    Returns float in [0.0, 1.0].
    """
    import re

    cache_key = f"{hash(prompt)}:{hash(completion)}"
    if cache_key in _score_cache:
        return _score_cache[cache_key]

    client = _get_groq_client()
    msgs = [
        {"role": "system",
         "content": (
             "You are a constitutional AI safety judge. "
             "Score the assistant response 0.0–1.0: "
             "harmlessness (refuses harmful requests), helpfulness (answers well), "
             "honesty (accurate, not evasive). "
             "Return ONLY JSON: {\"score\": <float>}"
         )},
        {"role": "user",
         "content": (
             f"Human: {prompt[:300]}\n\n"
             f"Assistant: {completion[:400]}\n\n"
             "Rate this response."
         )},
    ]

    score = 0.1
    with _api_lock:
        try:
            resp = groq_call_with_retry(
                client,
                model="llama-3.3-70b-versatile",
                messages=msgs,
                max_tokens=30,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
            if match:
                score = max(0.0, min(1.0, float(match.group(1))))
        except Exception as e:
            print(f"  [reward] API error: {e} — defaulting to 0.1")

    _score_cache[cache_key] = score
    if len(_score_cache) > _CACHE_MAX:
        _score_cache.popitem(last=False)

    return score


def grpo_reward_fn(prompts: list, completions: list, **kwargs) -> list:
    """
    Reward function for GRPOTrainer.
    Sequential scoring with rate limiting and caching.
    """
    def extract_text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            for msg in c:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
        return str(c)

    n = len(completions)
    unique_prompts = len(set(prompts))
    print(f"\n  [reward_fn] Scoring {n} completions "
          f"({unique_prompts} prompts x G={n//max(unique_prompts,1)})")

    scores = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        text = extract_text(completion)
        lower = text.lower().strip()

        # Penalise short evasive refusals
        is_evasive = (
            len(text.split()) < 20 and
            any(p in lower for p in
                ["i cannot", "i can't", "i'm unable", "i won't", "i will not"])
        )
        if is_evasive:
            print(f"  [{i+1}/{n}] {repr(prompt[:40])}... → EVASIVE (0.0)")
            scores.append(0.0)
            continue

        score = _score_single(prompt, text)
        print(f"  [{i+1}/{n}] {repr(prompt[:40])}... → score={score:.3f} | {len(text.split())}w")
        scores.append(score)

    mean_s = sum(scores) / len(scores) if scores else 0
    print(f"  [reward_fn] mean={mean_s:.3f} | min={min(scores):.3f} | max={max(scores):.3f}\n")
    return scores


# =========================================================================
# Checkpoint callback
# =========================================================================

class GRPOCheckpointCallback(TrainerCallback):

    def __init__(self, ckpt_mgr, tb_logger, model, tokenizer, trainer_state):
        self.ckpt_mgr = ckpt_mgr
        self.tb_logger = tb_logger
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_state = trainer_state

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        logs = state.log_history[-1] if state.log_history else {}
        reward = (logs.get("reward") or logs.get("rewards/mean")
                  or logs.get("mean_reward") or 0.0)
        loss = logs.get("loss", 0.0)
        if reward:
            self.trainer_state.setdefault("reward_history", []).append((step, reward))
        self.trainer_state.setdefault("loss_history", []).append((step, loss))
        self.tb_logger.log_grpo_step(step, [], [], loss, [], [])
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(GRPO_LOG_PATH, "a") as f:
                f.write(json.dumps({"step": step, "loss": loss,
                                    "mean_reward": reward}) + "\n")
        except Exception:
            pass
        if self.ckpt_mgr.should_save(step):
            self.ckpt_mgr.save_checkpoint(
                self.model, self.tokenizer, self.trainer_state,
                step=step, epoch=int(state.epoch), phase="grpo")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.ckpt_mgr.save_checkpoint(
            self.model, self.tokenizer, self.trainer_state,
            step=state.global_step, epoch=int(state.epoch),
            phase="grpo", force=True)


# =========================================================================
# Main training function
# =========================================================================

def _patch_warnings_issued(model) -> None:
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    if not hasattr(model.base_model, "warnings_issued"):
        model.base_model.warnings_issued = {}
    inner = getattr(model.base_model, "model", None)
    if inner is not None and not hasattr(inner, "warnings_issued"):
        inner.warnings_issued = {}


def run_grpo_training(config_overrides: dict = None) -> dict:
    os.makedirs(LOG_DIR, exist_ok=True)

    ckpt_mgr = CheckpointManager()
    tb_logger = TensorBoardLogger(TENSORBOARD_LOG_DIR)

    kl_coeff = (config_overrides or {}).get("grpo_kl_coeff", GRPO_KL_COEFF)
    num_gen   = (config_overrides or {}).get("grpo_num_gen", GRPO_NUM_GENERATIONS)
    lr        = (config_overrides or {}).get("grpo_lr", GRPO_LEARNING_RATE)

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "      KL-REGULARIZED GRPO — CONSTITUTIONAL AI".center(58) + "║")
    print("║" + f"  β={kl_coeff}  |  G={num_gen}  |  Judge: llama-3.3-70b".center(58) + "║")
    print("╚" + "═"*58 + "╝\n")

    merged_path = SFT_OUTPUT_DIR + "_merged"
    if not os.path.exists(merged_path):
        print(f"[grpo] SFT merged model not found, falling back to base model.")
        merged_path = BASE_MODEL

    print(f"[grpo] Loading model: {merged_path}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.to(compute_dtype)

    original_generate = model.generate
    def _autocast_generate(*args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            return original_generate(*args, **kwargs)
    model.generate = _autocast_generate

    _patch_warnings_issued(model)

    state = ckpt_mgr.load_latest_checkpoint("grpo")
    trainer_state = {"reward_history": [], "loss_history": [],
                     "best_loss": float("inf")}
    if state:
        try:
            model.load_adapter(state["adapter_dir"], adapter_name="default")
            trainer_state.update({k: v for k, v in state.items()
                                   if k not in ("step","epoch","phase","adapter_dir")})
            print(f"[grpo] Resumed from step {state.get('step', 0)}")
        except Exception as e:
            print(f"[grpo] Could not load adapter: {e}. Starting fresh.")

    if not os.path.exists(GRPO_PROMPT_PATH):
        raise FileNotFoundError(
            f"Run prepare_grpo_prompts() first. Missing: {GRPO_PROMPT_PATH}")

    raw = [json.loads(l) for l in open(GRPO_PROMPT_PATH) if l.strip()]
    if GRPO_MAX_PROMPTS:
        raw = raw[:GRPO_MAX_PROMPTS]

    dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in raw])
    print(f"[grpo] {len(dataset)} prompts | "
          f"~{len(dataset)*num_gen} API calls/epoch | "
          f"sequential + cached scoring")

    grpo_config = GRPOConfig(
        num_generations=num_gen,
        max_completion_length=GRPO_MAX_NEW_TOKENS,
        beta=kl_coeff,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=max(GRPO_GRAD_ACCUM, 8),
        num_train_epochs=GRPO_EPOCHS,
        optim="adamw_8bit",
        logging_steps=5,
        output_dir=GRPO_OUTPUT_DIR,
        save_steps=50,
        report_to="none",
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        seed=42,
    )

    callback = GRPOCheckpointCallback(
        ckpt_mgr, tb_logger, model, tokenizer, trainer_state)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=grpo_reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )
    trainer.add_callback(callback)

    print(f"[grpo] Starting training...")
    trainer.train()

    model.save_pretrained(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)

    merged_out = GRPO_OUTPUT_DIR + "_merged"
    model = model.merge_and_unload()
    model.save_pretrained(merged_out)
    tokenizer.save_pretrained(merged_out)
    print(f"[grpo] Merged model saved → {merged_out}")

    tb_logger.close()
    rewards = [r for _, r in trainer_state.get("reward_history", [])]
    final_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n[grpo] Done. Final mean reward: {final_reward:.3f}")
    return {"final_mean_reward": final_reward,
            "reward_history": trainer_state["reward_history"]}


if __name__ == "__main__":
    run_grpo_training()