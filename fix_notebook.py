"""
Fix LOCAL_QUICKSTART.ipynb:
1. Cell 1: use sys.executable -m pip instead of bare pip; remove sys.exit()
2. Cell 2: remove __file__ reference; add try/except for optional imports
3. Clear all old error outputs from all cells
4. Update kernelspec to constitutional-ai
"""
import json, copy

NB_PATH = "LOCAL_QUICKSTART.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ─── CELL 1 SOURCE ──────────────────────────────────────────────────────────
CELL1 = """\
# %% Cell 1 — Install Dependencies
# Uses sys.executable so pip always points to THIS venv's pip
import subprocess, sys, shutil

def run_pip(*args, **kwargs):
    \"\"\"Run pip inside the current venv and stream output.\"\"\"
    cmd = [sys.executable, "-m", "pip"] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace"
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print("STDERR:", result.stderr[:2000])
    return result.returncode == 0

print("=" * 70)
print("CONSTITUTIONAL AI - DEPENDENCY INSTALLATION")
print("=" * 70)

# ── GPU detection ──────────────────────────────────────────────────────────
has_nvidia = shutil.which("nvidia-smi") is not None
print(f"\\n🔍 GPU Detection: {'✅ NVIDIA GPU found' if has_nvidia else '⚠️  No NVIDIA GPU'}")
if not has_nvidia:
    print("\\n⚠️  No NVIDIA GPU found.")
    print("   Cells 1-6 (data prep) will still work on CPU.")
    print("   Cells 7-12 (training/inference) require an NVIDIA GPU.")

# ── Step 1: PyTorch with CUDA 12.4 ────────────────────────────────────────
print("\\n" + "=" * 70)
print("STEP 1: Checking / Installing PyTorch with CUDA 12.4")
print("=" * 70)

try:
    import torch
    print(f"✅ PyTorch already installed: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("Installing PyTorch 2.4.1 + CUDA 12.4 (~2.5 GB, takes a few minutes)...")
    ok = run_pip(
        "install",
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    )
    if not ok:
        print("❌ PyTorch installation failed! Check your internet connection.")
    else:
        print("✅ PyTorch installed successfully.")

# ── Step 2: transformers + training libs ──────────────────────────────────
print("\\n" + "=" * 70)
print("STEP 2: Installing transformers and training libraries")
print("=" * 70)
ok = run_pip(
    "install",
    "transformers==4.46.3", "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0", "datasets>=2.20.0",
    "peft>=0.12.0", "trl>=0.9.0"
)
if not ok:
    print("❌ Training libraries installation failed!")
else:
    print("✅ Training libraries installed.")

# ── Step 3: Unsloth ───────────────────────────────────────────────────────
print("\\n" + "=" * 70)
print("STEP 3: Installing Unsloth")
print("=" * 70)
ok = run_pip(
    "install",
    "unsloth[colab-new] @ https://github.com/unslothai/unsloth/archive/refs/heads/main.zip"
)
if not ok:
    print("⚠️  Unsloth git install failed — trying pip fallback...")
    ok = run_pip("install", "unsloth")
if ok:
    print("✅ Unsloth installed.")
else:
    print("❌ Unsloth installation failed. Training cells will not work.")

# ── Step 4: Remaining deps ────────────────────────────────────────────────
print("\\n" + "=" * 70)
print("STEP 4: Installing remaining dependencies")
print("=" * 70)
ok = run_pip(
    "install",
    "groq>=0.9.0", "python-dotenv", "pandas>=2.0.0",
    "pyarrow>=16.0.0", "requests", "streamlit>=1.35.0",
    "plotly>=5.22.0", "pyvis>=0.3.2", "networkx>=3.3",
    "tensorboard>=2.17.0", "sentencepiece>=0.2.1",
    "protobuf", "pyngrok"
)
if not ok:
    print("❌ Additional dependencies installation failed!")
else:
    print("✅ All additional dependencies installed.")

print("\\n" + "=" * 70)
print("✅ INSTALLATION COMPLETE — Please restart the kernel now!")
print("   Kernel menu → Restart Kernel (or press the ⟳ button)")
print("=" * 70)
""".splitlines(keepends=True)

# ─── CELL 2 SOURCE ──────────────────────────────────────────────────────────
CELL2 = """\
# %% Cell 2 — Verify Imports
import os, sys
from pathlib import Path

# ── Locate project directory (no __file__ in notebooks) ───────────────────
def find_project_dir():
    candidates = [
        Path.cwd(),
        Path.home() / "Downloads" / "Constitutional_AI",
        Path.home() / "OneDrive" / "Documents" / "Constitutional_AI",
    ]
    for c in candidates:
        if c.exists() and (c / "config.py").exists():
            return c
    return Path.cwd()

project_dir = find_project_dir()
os.chdir(project_dir)
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

print("=" * 70)
print("IMPORT VERIFICATION")
print("=" * 70)
print(f"📁 Project directory: {project_dir}")
print(f"🐍 Python executable: {sys.executable}")

# ── Core packages ──────────────────────────────────────────────────────────
errors = []
try:
    from groq import Groq
    import pandas as pd
    print(f"\\n✅ Core packages:")
    print(f"   - groq: OK")
    print(f"   - pandas: {pd.__version__}")
except ImportError as e:
    errors.append(f"Core import failed: {e}")
    print(f"\\n❌ Core import failed: {e}")
    print("   Run Cell 1 first, then restart the kernel.")

# ── PyTorch + CUDA ─────────────────────────────────────────────────────────
try:
    import torch
    print(f"\\n✅ PyTorch: {torch.__version__}")
    print(f"   - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - CUDA version: {torch.version.cuda}")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  CUDA not available — training cells will be skipped.")
        print("      If you have an NVIDIA GPU, reinstall PyTorch:")
        print("      pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124")
except ImportError as e:
    errors.append(f"PyTorch: {e}")
    print(f"\\n❌ PyTorch not installed: {e}")
    print("   Run Cell 1 and restart the kernel.")

# ── Training stack (optional — only needed for Cells 7-12) ────────────────
try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig, SFTTrainer
    import transformers, peft, datasets, trl
    print(f"\\n✅ Training stack:")
    print(f"   - unsloth: OK")
    print(f"   - transformers: {transformers.__version__}")
    print(f"   - trl: {trl.__version__}")
    print(f"   - peft: {peft.__version__}")
    print(f"   - datasets: {datasets.__version__}")
except ImportError as e:
    print(f"\\n⚠️  Training stack not fully installed: {e}")
    print("   This is OK if you only need dataset prep (Cells 4-6).")
    print("   Run Cell 1 and restart kernel to install training libs.")

if not errors:
    print("\\n" + "=" * 70)
    print("✅ CORE IMPORTS OK — Ready to run data preparation cells.")
    print("=" * 70)
else:
    print("\\n" + "=" * 70)
    print("⚠️  Some imports failed — run Cell 1 and restart the kernel.")
    print("=" * 70)
""".splitlines(keepends=True)

# ─── Patch cells ─────────────────────────────────────────────────────────────
code_cell_index = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    code_cell_index += 1
    # Clear all old outputs
    cell["outputs"] = []
    cell["execution_count"] = None

    if code_cell_index == 1:
        cell["source"] = CELL1
    elif code_cell_index == 2:
        cell["source"] = CELL2

# ─── Update kernelspec ────────────────────────────────────────────────────────
nb["metadata"]["kernelspec"] = {
    "display_name": "Constitutional AI (3.12)",
    "language": "python",
    "name": "constitutional-ai"
}
nb["metadata"]["language_info"]["version"] = "3.12.7"

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook fixed and saved.")
print("   - Cell 1: uses sys.executable -m pip (no bare 'pip')")
print("   - Cell 2: removed __file__, CUDA check is non-fatal warning")
print("   - All old error outputs cleared")
print("   - Kernelspec updated to 'constitutional-ai' (Python 3.12)")
