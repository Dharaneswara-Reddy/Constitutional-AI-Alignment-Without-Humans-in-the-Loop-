"""
Phase 1: Supervised Fine-Tuning — Constitutional AI.
Paper: Bai et al. 2022, Section 3.

Method:
  1. Base model: Qwen2-0.5B-Instruct (helpful-only baseline)
  2. SFT on critique+revision pairs from HH-RLHF (harmlessness)
  3. Mixed with direct chosen responses (helpfulness, Section 3.2)
  4. This SFT model becomes pi_ref for GRPO Phase 2.
"""

import json, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (BASE_MODEL, MAX_SEQ_LENGTH, LOAD_IN_4BIT,
                    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
                    SFT_LEARNING_RATE, SFT_BATCH_SIZE, SFT_GRAD_ACCUM,
                    SFT_EPOCHS, SFT_WARMUP_STEPS, SFT_OPTIMIZER,
                    SFT_OUTPUT_DIR, SFT_DATA_PATH, TENSORBOARD_LOG_DIR,
                    SFT_LOG_PATH, LOG_DIR)
from src.training.checkpoint_manager import CheckpointManager
from src.visualization.tensorboard_logger import TensorBoardLogger


def load_or_resume_model(ckpt_mgr: CheckpointManager):
    """Loads model; resumes LoRA adapter from checkpoint if available."""
    from unsloth import FastLanguageModel
    state = ckpt_mgr.load_latest_checkpoint("sft")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=LOAD_IN_4BIT)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK, target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42)
    start_epoch, start_step = 0, 0
    trainer_state = {"loss_history": [], "best_loss": float("inf")}
    if state:
        try:
            from peft import PeftModel
            model.load_adapter(state["adapter_dir"], adapter_name="default")
            start_epoch = state.get("epoch", 0)
            start_step = state.get("step", 0)
            trainer_state = {k: v for k, v in state.items()
                             if k not in ("step", "epoch", "phase", "adapter_dir")}
        except Exception as e:
            print(f"[sft] Could not load adapter: {e}. Starting fresh.")
    return model, tokenizer, start_epoch, start_step, trainer_state


def format_sft_sample(example: dict, tokenizer) -> dict:
    """Formats (prompt, revision) using Qwen2 chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
        {"role": "user", "content": example.get("prompt", "")},
        {"role": "assistant", "content": example.get("revision", "")}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


class CAISFTCallback:
    """Custom TRL callback for checkpoint saving and TensorBoard logging."""
    def __init__(self, ckpt_mgr: CheckpointManager, tb_logger: TensorBoardLogger,
                 model, tokenizer, trainer_state: dict):
        self.ckpt_mgr = ckpt_mgr
        self.tb_logger = tb_logger
        self.model = model
        self.tokenizer = tokenizer
        self.trainer_state = trainer_state

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if state.log_history:
            metrics = state.log_history[-1]
            loss = metrics.get("loss", None)
            lr = metrics.get("learning_rate", None)
            if loss:
                self.trainer_state["loss_history"].append((step, loss))
                if loss < self.trainer_state.get("best_loss", float("inf")):
                    self.trainer_state["best_loss"] = loss
            self.tb_logger.log_sft_step(step, metrics)
            # JSON log for dashboard
            try:
                with open(SFT_LOG_PATH, "a") as f:
                    f.write(json.dumps({"step": step, "loss": loss, "lr": lr}) + "\n")
            except Exception:
                pass
        if self.ckpt_mgr.should_save(step):
            self.ckpt_mgr.save_checkpoint(
                self.model, self.tokenizer, self.trainer_state,
                step=step, epoch=int(state.epoch), phase="sft")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.ckpt_mgr.save_checkpoint(
            self.model, self.tokenizer, self.trainer_state,
            step=state.global_step, epoch=int(state.epoch), phase="sft", force=True)


def run_sft_training(config_overrides: dict = None) -> dict:
    """
    Main SFT training function. Supports resume from checkpoint.
    config_overrides: optional dict from Streamlit dashboard sliders.
    Returns training metrics dict.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    os.makedirs(LOG_DIR, exist_ok=True)

    ckpt_mgr = CheckpointManager()
    tb_logger = TensorBoardLogger(TENSORBOARD_LOG_DIR)

    # Config overrides from dashboard
    lr = (config_overrides or {}).get("sft_lr", SFT_LEARNING_RATE)
    epochs = (config_overrides or {}).get("sft_epochs", SFT_EPOCHS)
    batch_size = (config_overrides or {}).get("sft_batch_size", SFT_BATCH_SIZE)

    print("=" * 60)
    print("Phase 1: SFT Training")
    print(f"  Model:   {BASE_MODEL}")
    print(f"  LR:      {lr}  |  Epochs: {epochs}  |  Batch: {batch_size}")
    print("=" * 60)

    model, tokenizer, start_epoch, start_step, trainer_state = load_or_resume_model(ckpt_mgr)

    # Load SFT dataset
    if not os.path.exists(SFT_DATA_PATH):
        raise FileNotFoundError(f"Run prepare_datasets.py first. Missing: {SFT_DATA_PATH}")
    raw_data = [json.loads(l) for l in open(SFT_DATA_PATH) if l.strip()]
    print(f"[sft] Loaded {len(raw_data)} training samples")

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(lambda ex: format_sft_sample(ex, tokenizer))

    callback = CAISFTCallback(ckpt_mgr, tb_logger, model, tokenizer, trainer_state)

    training_args = TrainingArguments(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        warmup_steps=SFT_WARMUP_STEPS,
        learning_rate=lr,
        optim=SFT_OPTIMIZER,
        logging_steps=1,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
    )

    # NOTE: TRL 0.24+ uses processing_class= instead of tokenizer=
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    # Monkey-patch the callback
    class _TRLCallback:
        def on_step_end(self, args, state, control, **kwargs):
            callback.on_step_end(args, state, control, **kwargs)
        def on_epoch_end(self, args, state, control, **kwargs):
            callback.on_epoch_end(args, state, control, **kwargs)

    from transformers import TrainerCallback
    class _Cb(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            callback.on_step_end(args, state, control, **kwargs)
        def on_epoch_end(self, args, state, control, **kwargs):
            callback.on_epoch_end(args, state, control, **kwargs)

    trainer.add_callback(_Cb())
    trainer.train()

    # Save merged 16-bit model — this is pi_ref for GRPO
    merged_dir = SFT_OUTPUT_DIR + "_merged"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[sft] Merged 16-bit π_ref saved → {merged_dir}")

    tb_logger.close()
    final_loss = trainer_state["loss_history"][-1][1] if trainer_state["loss_history"] else 0
    print(f"[sft] SFT complete. Final loss: {final_loss:.4f}")
    print("[sft] This model is now π_ref for GRPO Phase 2")
    return {"final_loss": final_loss, "loss_history": trainer_state["loss_history"]}


if __name__ == "__main__":
    run_sft_training()
