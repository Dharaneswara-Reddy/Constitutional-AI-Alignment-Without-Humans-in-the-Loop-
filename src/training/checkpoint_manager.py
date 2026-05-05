"""
Checkpoint Manager — Save/resume for interrupted Google Colab training.
Colab free tier disconnects after ~12hrs. This saves complete state every N steps.

Saves:
  - LoRA adapter weights
  - Trainer state (step, epoch, loss/reward history)
  - Random states (torch, numpy, python) for reproducibility
  - Config snapshot

Auto-detects Google Drive mount and copies checkpoints there for persistence.
"""

import os, json, pickle, shutil, glob
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHECKPOINT_DIR, CHECKPOINT_EVERY_N_STEPS


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR,
                 save_every_n_steps: int = CHECKPOINT_EVERY_N_STEPS):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self.gdrive_dir = None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._detect_google_drive()

    def _detect_google_drive(self):
        """Check if running in Colab with Google Drive mounted."""
        if os.path.exists("/content/drive/MyDrive"):
            self.gdrive_dir = Path("/content/drive/MyDrive/cai_checkpoints")
            self.gdrive_dir.mkdir(parents=True, exist_ok=True)
            print("[ckpt] Google Drive detected — checkpoints also save to Drive")

    def _checkpoint_path(self, phase: str, epoch: int, step: int) -> Path:
        return self.checkpoint_dir / f"{phase}_epoch{epoch}_step{step}"

    def save_checkpoint(self, model, tokenizer, trainer_state: dict,
                        step: int, epoch: int, phase: str, force: bool = False):
        """
        Saves complete training state to disk (and Google Drive if available).
        Keeps only the last 3 checkpoints to save disk space.

        trainer_state should contain:
          loss_history, reward_history, kl_history, best_loss, completed_prompts
        """
        import torch, numpy as np

        ckpt_path = self._checkpoint_path(phase, epoch, step)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        adapter_dir = ckpt_path / "adapter_model"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # Save trainer state
        state_to_save = {
            "step": step,
            "epoch": epoch,
            "phase": phase,
            **trainer_state
        }
        with open(ckpt_path / "trainer_state.json", "w") as f:
            json.dump(state_to_save, f, indent=2)

        # Save random states for exact reproducibility
        try:
            random_states = {
                "torch": torch.get_rng_state().numpy().tolist(),
                "torch_cuda": torch.cuda.get_rng_state().numpy().tolist()
                    if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            }
            with open(ckpt_path / "random_states.pkl", "wb") as f:
                pickle.dump(random_states, f)
        except Exception as e:
            print(f"[ckpt] Warning: could not save random states: {e}")

        # Save config snapshot
        try:
            import config as cfg
            cfg_dict = {k: v for k, v in vars(cfg).items()
                        if not k.startswith("_") and isinstance(v, (int, float, str, bool, list))}
            with open(ckpt_path / "config_snapshot.json", "w") as f:
                json.dump(cfg_dict, f, indent=2)
        except Exception:
            pass

        print(f"[ckpt] Saved: {ckpt_path.name}")

        # Copy to Google Drive if available
        if self.gdrive_dir:
            try:
                gdrive_ckpt = self.gdrive_dir / ckpt_path.name
                shutil.copytree(ckpt_path, gdrive_ckpt, dirs_exist_ok=True)
                print(f"[ckpt] Also saved to Google Drive: {gdrive_ckpt}")
            except Exception as e:
                print(f"[ckpt] Drive copy warning: {e}")

        # Prune old checkpoints (keep last 3)
        self._prune_old_checkpoints(phase, keep=3)

    def _prune_old_checkpoints(self, phase: str, keep: int = 3):
        """Remove oldest checkpoints beyond the keep limit."""
        pattern = str(self.checkpoint_dir / f"{phase}_epoch*_step*")
        existing = sorted(glob.glob(pattern), key=os.path.getmtime)
        for old in existing[:-keep]:
            try:
                shutil.rmtree(old)
                print(f"[ckpt] Pruned old checkpoint: {Path(old).name}")
            except Exception:
                pass

    def load_latest_checkpoint(self, phase: str) -> dict | None:
        """
        Finds and loads the most recent checkpoint for the given phase.
        Returns trainer_state dict or None if no checkpoint exists.
        """
        pattern = str(self.checkpoint_dir / f"{phase}_epoch*_step*")
        existing = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not existing:
            print(f"[ckpt] No checkpoint found for phase='{phase}' — starting fresh")
            return None

        latest = Path(existing[-1])
        state_file = latest / "trainer_state.json"
        if not state_file.exists():
            return None

        with open(state_file) as f:
            state = json.load(f)

        best_loss = state.get("best_loss", float("inf"))
        print(f"[ckpt] Resuming from epoch {state['epoch']}, step {state['step']} "
              f"(loss: {best_loss:.4f})")
        state["adapter_dir"] = str(latest / "adapter_model")
        return state

    def should_save(self, current_step: int) -> bool:
        return current_step % self.save_every_n_steps == 0

    def get_checkpoint_summary(self) -> dict:
        """Returns dict with all checkpoint info for dashboard display."""
        summary = {"phases": {}, "total_checkpoints": 0}
        for phase in ["sft", "grpo"]:
            pattern = str(self.checkpoint_dir / f"{phase}_epoch*_step*")
            existing = sorted(glob.glob(pattern), key=os.path.getmtime)
            if existing:
                latest = Path(existing[-1])
                state_file = latest / "trainer_state.json"
                state = {}
                if state_file.exists():
                    with open(state_file) as f:
                        state = json.load(f)
                summary["phases"][phase] = {
                    "count": len(existing),
                    "latest": latest.name,
                    "step": state.get("step", 0),
                    "epoch": state.get("epoch", 0),
                    "best_loss": state.get("best_loss", None),
                }
            else:
                summary["phases"][phase] = {"count": 0, "latest": None}
            summary["total_checkpoints"] += summary["phases"][phase]["count"]
        summary["gdrive_available"] = self.gdrive_dir is not None
        return summary
