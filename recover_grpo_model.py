"""
recover_grpo_model.py
=====================
Run this after a GRPO training crash to save the final model from the
latest checkpoint.

The GRPO training loop completed (checkpoint grpo_epoch1_step250 exists),
but the post-training save / merge_and_unload() crashed before it could write
the weights.  This script:

  1. Finds the most recent GRPO checkpoint automatically.
  2. Loads the base (SFT-merged) model + the LoRA adapter from the checkpoint.
  3. Saves the LoRA adapter to   outputs/grpo_model          (same as normal run)
  4. Merges & saves the full model to  outputs/grpo_model_merged  (same as normal run)

Usage:
    .venv\Scripts\python.exe recover_grpo_model.py
"""

import os, sys, glob, json, torch
from pathlib import Path

# ── make sure project root is on the path ──────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    SFT_OUTPUT_DIR, GRPO_OUTPUT_DIR, BASE_MODEL,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    CHECKPOINT_DIR,
)

# ── monkey-patch for older PyTorch (< 2.5) + transformers 5.5 ──────────────
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        if target == "":
            raise ValueError("Cannot set an empty string as a submodule name.")
        atoms = target.split(".")
        name = atoms[-1]
        parent = self.get_submodule(".".join(atoms[:-1])) if len(atoms) > 1 else self
        setattr(parent, name, module)
    torch.nn.Module.set_submodule = _set_submodule


def find_latest_grpo_checkpoint() -> Path | None:
    """Return the most recently modified grpo checkpoint directory."""
    pattern = str(Path(CHECKPOINT_DIR) / "grpo_epoch*_step*")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        return None
    return Path(candidates[-1])


def main():
    print("\n" + "=" * 60)
    print("  GRPO MODEL RECOVERY")
    print("=" * 60)

    # ── 1. Locate checkpoint ───────────────────────────────────────────────
    ckpt_dir = find_latest_grpo_checkpoint()
    if ckpt_dir is None:
        print("[ERROR] No GRPO checkpoint found in", CHECKPOINT_DIR)
        sys.exit(1)

    adapter_dir = ckpt_dir / "adapter_model"
    if not adapter_dir.exists():
        print(f"[ERROR] Adapter not found at {adapter_dir}")
        sys.exit(1)

    state_file = ckpt_dir / "trainer_state.json"
    step = epoch = "?"
    if state_file.exists():
        with open(state_file) as f:
            st = json.load(f)
        step = st.get("step", "?")
        epoch = st.get("epoch", "?")

    print(f"[recover] Using checkpoint : {ckpt_dir.name}")
    print(f"[recover] Epoch={epoch}  Step={step}")
    print(f"[recover] Adapter path    : {adapter_dir}")

    # ── 2. Choose base model ───────────────────────────────────────────────
    merged_sft = SFT_OUTPUT_DIR + "_merged"
    if os.path.exists(merged_sft):
        base_path = merged_sft
        print(f"[recover] Base model: SFT-merged  ({merged_sft})")
    else:
        base_path = BASE_MODEL
        print(f"[recover] SFT-merged not found — using HF base: {BASE_MODEL}")

    # ── 3. Load base model ─────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Always load in plain float16 (no 4-bit/bitsandbytes) for recovery.
    # 4-bit models cannot be cast with .to() or saved after merge_and_unload().
    print(f"\n[recover] Loading base model in float16 (no quantization) …")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.float16,
        device_map="cpu",       # CPU to avoid GPU OOM during merge
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 4. Load LoRA adapter ───────────────────────────────────────────────
    print(f"[recover] Loading LoRA adapter from checkpoint …")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    # ── 5. Save LoRA model (adapter only) → outputs/grpo_model ────────────
    lora_out = GRPO_OUTPUT_DIR
    os.makedirs(lora_out, exist_ok=True)
    print(f"\n[recover] Saving LoRA adapter → {lora_out}")
    model.save_pretrained(lora_out)
    tokenizer.save_pretrained(lora_out)
    print(f"[recover] ✓ LoRA adapter saved.")

    # ── 6. Merge & save full model → outputs/grpo_model_merged ────────────
    merged_out = GRPO_OUTPUT_DIR + "_merged"
    os.makedirs(merged_out, exist_ok=True)
    print(f"\n[recover] Merging LoRA weights into base model …")
    # merge_and_unload() is safe here because the base model is plain float16
    # (not 4-bit), so no bitsandbytes constraints apply.
    merged_model = model.merge_and_unload()
    print(f"[recover] Saving merged model → {merged_out}")
    merged_model.save_pretrained(merged_out, safe_serialization=True)
    tokenizer.save_pretrained(merged_out)
    print(f"[recover] ✓ Merged model saved.")

    # ── 7. Quick sanity check (CPU inference) ─────────────────────────────
    print("\n[recover] Sanity-checking merged model output …")
    merged_model.eval()
    inputs = tokenizer(
        "Is it okay to steal food if you are very hungry?",
        return_tensors="pt",
    )  # already on CPU
    with torch.no_grad():
        out = merged_model.generate(**inputs, max_new_tokens=80, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n  Prompt : Is it okay to steal food if you are very hungry?")
    print(f"  Response: {response.strip()}\n")

    print("=" * 60)
    print("  RECOVERY COMPLETE")
    print(f"  LoRA adapter : {lora_out}")
    print(f"  Merged model : {merged_out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
