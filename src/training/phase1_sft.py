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
from transformers import DataCollatorForSeq2Seq

# --- MONKEY PATCH FOR TRANSFORMERS 5.5.0 + PYTORCH < 2.5 ---
# PyTorch < 2.5 does not have set_submodule, but transformers 5.5.0+ requires it.
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig

    state = ckpt_mgr.load_latest_checkpoint("sft")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if LOAD_IN_4BIT else None

    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False  # Required for gradient checkpointing

    start_epoch, start_step = 0, 0
    trainer_state = {"loss_history": [], "best_loss": float("inf")}
    if state:
        try:
            model.load_adapter(state["adapter_dir"], adapter_name="default")
            start_epoch = state.get("epoch", 0)
            start_step = state.get("step", 0)
            trainer_state = {k: v for k, v in state.items()
                             if k not in ("step", "epoch", "phase", "adapter_dir")}
        except Exception as e:
            print(f"[sft] Could not load adapter: {e}. Starting fresh.")
    return model, tokenizer, start_epoch, start_step, trainer_state


def format_sft_sample(example: dict, tokenizer) -> str:
    """Formats (prompt, revision) using Qwen2 chat template. Returns text string."""
    messages = [
        {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
        {"role": "user", "content": example.get("prompt", "")},
        {"role": "assistant", "content": example.get("revision", "")}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


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
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, TrainerCallback

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

    # ── Pre-tokenize dataset to bypass TRL's internal tokenize_fn ──────────
    # TRL 0.24's SFTTrainer internally looks for 'prompt'+'completion' columns.
    # By pre-tokenizing and passing input_ids+labels directly, we skip that path.
    def tokenize_sample(example):
        text = format_sft_sample(example, tokenizer)
        result = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        tokenize_sample,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    print(f"[sft] Tokenized {len(dataset)} samples")

    callback = CAISFTCallback(ckpt_mgr, tb_logger, model, tokenizer, trainer_state)

    training_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=SFT_WARMUP_STEPS,
        learning_rate=lr,
        optim=SFT_OPTIMIZER,
        logging_steps=1,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        torch_empty_cache_steps=10,
    )

    # Data collator handles padding within batches
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # SFTTrainer with pre-tokenized dataset — no internal processing needed
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        args=training_args,
    )

    class _Cb(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            callback.on_step_end(args, state, control, **kwargs)
        def on_epoch_end(self, args, state, control, **kwargs):
            callback.on_epoch_end(args, state, control, **kwargs)

    trainer.add_callback(_Cb())
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
    print(f"[sft] LoRA adapter saved → {SFT_OUTPUT_DIR}")

    # Save merged 16-bit model — this is pi_ref for GRPO
    merged_dir = SFT_OUTPUT_DIR + "_merged"
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[sft] Merged 16-bit π_ref saved → {merged_dir}")

    tb_logger.close()
    final_loss = trainer_state["loss_history"][-1][1] if trainer_state["loss_history"] else 0
    print(f"[sft] SFT complete. Final loss: {final_loss:.4f}")
    print("[sft] This model is now π_ref for GRPO Phase 2")
    return {"final_loss": final_loss, "loss_history": trainer_state["loss_history"]}


if __name__ == "__main__":
    run_sft_training()