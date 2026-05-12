#!/usr/bin/env python
"""Test that all imports work without unsloth."""

print("Testing imports...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print("✅ transformers imports OK")
except Exception as e:
    print(f"❌ transformers import failed: {e}")
    exit(1)

try:
    from peft import get_peft_model, LoraConfig
    print("✅ peft imports OK")
except Exception as e:
    print(f"❌ peft import failed: {e}")
    exit(1)

try:
    from trl import SFTTrainer, GRPOTrainer, GRPOConfig
    print("✅ trl imports OK")
except Exception as e:
    print(f"❌ trl import failed: {e}")
    exit(1)

try:
    import torch
    print(f"✅ torch {torch.__version__} OK")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ torch import failed: {e}")
    exit(1)

print("\n✅ ALL IMPORTS SUCCESSFUL - Ready to train!")
