"""
Phase 2: KL-Regularized GRPO — Constitutional AI RL Phase.
Paper: Bai et al. 2022, Section 4.

GRPO replaces PPO's critic network with group-relative reward normalization:
  For G responses per prompt, advantage_i = (r_i - mean) / std

Anti-reward-hacking (paper Sections 4.3 + 4.4):
  - Principle ensembling: 4 random RL principles per score
  - Soft labels clamped to [0.4, 0.6]
  - Evasiveness penalty
  - KL coefficient beta=0.1 (mathematical leash on policy)

NOTE: TRL 0.24+ uses beta= (not kl_coeff=) and max_completion_length= (not max_new_tokens=)
      and processing_class= (not tokenizer=)
"""

import json, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (SFT_OUTPUT_DIR, GRPO_OUTPUT_DIR, GRPO_PROMPT_PATH,
                    MAX_SEQ_LENGTH, LOAD_IN_4BIT,
                    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
                    GRPO_NUM_GENERATIONS, GRPO_MAX_NEW_TOKENS, GRPO_LEARNING_RATE,
                    GRPO_BATCH_SIZE, GRPO_GRAD_ACCUM, GRPO_EPOCHS,
                    GRPO_KL_COEFF,  # β — KL penalty: R_KL = score - β*KL(π_θ||π_ref)
                    GRPO_MAX_PROMPTS, TENSORBOARD_LOG_DIR, GRPO_LOG_PATH, LOG_DIR)
from src.training.checkpoint_manager import CheckpointManager
from src.visualization.tensorboard_logger import TensorBoardLogger
from src.training.reward_function import constitutional_reward_fn

# Unsloth patches GRPOTrainer automatically when FastLanguageModel is imported.
# No manual patching needed in unsloth >= 2024.12 — optimizations are applied
# during model loading via FastLanguageModel.get_peft_model().
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from datasets import Dataset


class GRPOCheckpointCallback(TrainerCallback):
    """Saves every N steps + TensorBoard logging — critical for Colab resume."""

    def __init__(self, ckpt_mgr: CheckpointManager, tb_logger: TensorBoardLogger,
                 model, tokenizer, trainer_state: dict):
        self.ckpt_mgr = ckpt_mgr
        self.tb_logger = tb_logger
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_state = trainer_state

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        logs = state.log_history[-1] if state.log_history else {}
        reward_scores = kwargs.get("reward_scores", [])
        kl_values = kwargs.get("kl_values", [])
        loss = logs.get("loss", 0.0)

        if reward_scores:
            self.trainer_state.setdefault("reward_history", []).append(
                (step, sum(reward_scores)/len(reward_scores)))
        self.trainer_state.setdefault("loss_history", []).append((step, loss))

        # TensorBoard logging
        self.tb_logger.log_grpo_step(
            step, reward_scores or [], kl_values or [], loss, [], [])

        # JSON log for dashboard
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(GRPO_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "loss": loss,
                    "mean_reward": sum(reward_scores)/len(reward_scores) if reward_scores else None
                }) + "\n")
        except Exception:
            pass

        if self.ckpt_mgr.should_save(step):
            self.ckpt_mgr.save_checkpoint(
                self.model, self.tokenizer, self.trainer_state,
                step=step, epoch=int(state.epoch), phase="grpo")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.ckpt_mgr.save_checkpoint(
            self.model, self.tokenizer, self.trainer_state,
            step=state.global_step, epoch=int(state.epoch), phase="grpo", force=True)


def run_grpo_training(config_overrides: dict = None) -> dict:
    """
    Main GRPO training function with resume support.
    Returns comprehensive metrics dict.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    ckpt_mgr = CheckpointManager()
    tb_logger = TensorBoardLogger(TENSORBOARD_LOG_DIR)

    kl_coeff = (config_overrides or {}).get("grpo_kl_coeff", GRPO_KL_COEFF)
    num_gen = (config_overrides or {}).get("grpo_num_gen", GRPO_NUM_GENERATIONS)
    lr = (config_overrides or {}).get("grpo_lr", GRPO_LEARNING_RATE)

    # Training banner
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "       KL-REGULARIZED GRPO — CONSTITUTIONAL AI".center(58) + "║")
    print("║" + f"  Policy: π_θ (trainable)  |  Reference: π_ref (frozen)".center(58) + "║")
    print("║" + f"  β (KL coeff): {kl_coeff}        |  Group size G: {num_gen}".center(58) + "║")
    print("║" + "  Judge: llama-3.3-70b via Groq API".center(58) + "║")
    print("║" + "  Reward: constitutional_score - β*KL(π_θ||π_ref)".center(58) + "║")
    print("╚" + "═"*58 + "╝\n")

    # Load SFT merged model as pi_ref AND starting point for pi_theta
    merged_path = SFT_OUTPUT_DIR + "_merged"
    if not os.path.exists(merged_path):
        print(f"[grpo] SFT merged model not found at {merged_path}")
        print("[grpo] Falling back to base model...")
        merged_path = "unsloth/Qwen2-0.5B-Instruct-bnb-4bit"

    print(f"[grpo] Loading model: {merged_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merged_path, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=LOAD_IN_4BIT)

    # Re-inject LoRA — trainable pi_theta on top of frozen pi_ref
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK, target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42)

    # Check for existing checkpoint
    state = ckpt_mgr.load_latest_checkpoint("grpo")
    trainer_state = {"reward_history": [], "loss_history": [], "best_loss": float("inf")}
    if state:
        try:
            model.load_adapter(state["adapter_dir"], adapter_name="default")
            trainer_state.update({k: v for k, v in state.items()
                                   if k not in ("step","epoch","phase","adapter_dir")})
        except Exception as e:
            print(f"[grpo] Could not load adapter: {e}. Starting fresh.")

    # Load GRPO prompt dataset
    if not os.path.exists(GRPO_PROMPT_PATH):
        raise FileNotFoundError(f"Run prepare_grpo_prompts() first. Missing: {GRPO_PROMPT_PATH}")
    raw = [json.loads(l) for l in open(GRPO_PROMPT_PATH) if l.strip()]
    dataset = Dataset.from_list(raw)
    dataset = dataset.select_columns(["prompt"])  # GRPO only needs prompts
    print(f"[grpo] Loaded {len(dataset)} prompts for GRPO training")

    # NOTE: TRL 0.24+ parameter names:
    #   beta= (was kl_coeff=), max_completion_length= (was max_new_tokens=)
    grpo_config = GRPOConfig(
        num_generations=num_gen,                     # G — group size
        max_completion_length=GRPO_MAX_NEW_TOKENS,   # TRL 0.24+: was max_new_tokens=
        beta=kl_coeff,                               # β — KL penalty: R_KL = score - β*KL(π_θ||π_ref)
                                                     # TRL 0.24+: was kl_coeff=
        learning_rate=lr,
        per_device_train_batch_size=GRPO_BATCH_SIZE,
        gradient_accumulation_steps=GRPO_GRAD_ACCUM,
        num_train_epochs=GRPO_EPOCHS,
        optim="adamw_8bit",
        logging_steps=5,
        output_dir=GRPO_OUTPUT_DIR,
        save_steps=50,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=42,
    )

    callback = GRPOCheckpointCallback(ckpt_mgr, tb_logger, model, tokenizer, trainer_state)

    # NOTE: TRL 0.24+ uses processing_class= instead of tokenizer=
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,          # TRL 0.24+: was tokenizer=
        reward_funcs=constitutional_reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        # ref_model=None — Unsloth uses frozen base weights as π_ref internally
    )
    trainer.add_callback(callback)

    print(f"[grpo] Starting GRPO training: {len(dataset)} prompts, G={num_gen}, β={kl_coeff}")
    trainer.train()

    # Save final model
    model.save_pretrained(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
    print(f"[grpo] LoRA adapter saved: {GRPO_OUTPUT_DIR}")

    model.save_pretrained_merged(GRPO_OUTPUT_DIR + "_merged", tokenizer, save_method="merged_16bit")
    print(f"[grpo] Merged 16-bit saved: {GRPO_OUTPUT_DIR}_merged")

    try:
        model.save_pretrained_gguf(GRPO_OUTPUT_DIR + "_gguf", tokenizer, quantization_method="q4_k_m")
        print(f"[grpo] GGUF Q4_K_M saved: {GRPO_OUTPUT_DIR}_gguf")
    except Exception as e:
        print(f"[grpo] GGUF export skipped: {e}")

    tb_logger.close()

    rewards = [r for _, r in trainer_state.get("reward_history", [])]
    final_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"\n[grpo] GRPO training complete. Final mean reward: {final_reward:.3f}")
    print("[grpo] KL-regularized aligned model saved.")
    return {"final_mean_reward": final_reward, "reward_history": trainer_state["reward_history"]}


if __name__ == "__main__":
    run_grpo_training()
