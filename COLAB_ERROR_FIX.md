# 🚨 Colab ImportError Fix

If you see this error in Google Colab:
```
ImportError: cannot import name '_unsloth_get_mm_token_id'
from 'unsloth_zoo.rl_replacements'
```

**Paste these two code blocks into Colab cells and follow the steps exactly.**

---

## WHY this happens

Colab pre-installs an old `unsloth_zoo`. When `unsloth` tries to import,
it expects a new symbol `_unsloth_get_mm_token_id` that the old `unsloth_zoo`
doesn't have yet. The fix is to completely uninstall the old versions and
reinstall both packages fresh so pip resolves compatible versions.

---

## STEP 1 — Paste this into a new cell and run it

```python
import subprocess, sys
pip = [sys.executable, "-m", "pip"]

print("[1/4] Removing old conflicting packages...")
subprocess.run(pip + ["uninstall", "-y", "unsloth", "unsloth-zoo", "unsloth_zoo"],
               capture_output=True)

print("[2/4] Installing unsloth + unsloth_zoo together (compatible versions)...")
subprocess.run(pip + ["install", "--quiet", "unsloth", "unsloth_zoo"], check=True)

print("[3/4] Installing TRL stack...")
subprocess.run(pip + ["install", "--quiet",
    "trl", "transformers>=4.44.0", "peft>=0.12.0",
    "datasets>=2.20.0", "accelerate>=0.30.0", "bitsandbytes>=0.43.0"], check=True)

print("[4/4] Installing project deps...")
subprocess.run(pip + ["install", "--quiet",
    "groq>=0.9.0", "streamlit", "plotly", "pyvis", "networkx",
    "tensorboard", "pandas", "pyarrow", "sentencepiece", "protobuf", "pyngrok"], check=True)

print()
print("=" * 50)
print(" DONE. Now: Runtime → Restart runtime")
print(" After restart, run STEP 2 below.")
print("=" * 50)
```

## STEP 2 — After restarting runtime, paste this verify cell

```python
# Run this AFTER Runtime → Restart runtime
from unsloth import FastLanguageModel          # DO NOT import PatchGRPOTrainer — it was removed
from trl import GRPOTrainer, GRPOConfig, SFTTrainer
from groq import Groq
import trl, transformers, peft

print("✅ All imports OK")
print(f"   trl: {trl.__version__}")
print(f"   transformers: {transformers.__version__}")
```

---

## Key rules to avoid this error

| ❌ Wrong | ✅ Correct |
|---|---|
| `from unsloth import FastLanguageModel, PatchGRPOTrainer` | `from unsloth import FastLanguageModel` |
| `PatchGRPOTrainer()` | *(delete this line — no longer needed)* |
| `pip install --force-reinstall unsloth` | `pip uninstall -y unsloth unsloth_zoo && pip install unsloth unsloth_zoo` |
| Running imports without restarting after install | Always restart runtime after Cell 1 |

## Instead of `colab_notebook.ipynb`, use `COLAB_QUICKSTART.ipynb`

The file `COLAB_QUICKSTART.ipynb` in your project folder has all of this
already baked in. Upload that notebook to Colab instead of the old one.
