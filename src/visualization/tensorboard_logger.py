"""
TensorBoard Logger — Comprehensive training visualization.
Logs SFT and GRPO metrics to TensorBoard event files.
Also computes and logs alignment metrics after evaluation.
"""

import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TENSORBOARD_LOG_DIR, GRPO_KL_COEFF


class TensorBoardLogger:
    def __init__(self, log_dir: str = TENSORBOARD_LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self._ok = True
        except ImportError:
            print("[tb] tensorboard not installed — logging disabled")
            self.writer = None
            self._ok = False

    def _w(self, tag, val, step):
        if self._ok and self.writer and val is not None:
            try:
                self.writer.add_scalar(tag, float(val), step)
            except Exception:
                pass

    def log_sft_step(self, step: int, metrics: dict):
        """Log SFT training metrics per step."""
        self._w("train/loss", metrics.get("loss"), step)
        self._w("train/learning_rate", metrics.get("learning_rate"), step)
        self._w("train/epoch", metrics.get("epoch"), step)
        self._w("train/grad_norm", metrics.get("grad_norm"), step)
        self._w("eval/loss", metrics.get("eval_loss"), step)

    def log_grpo_step(self, step: int, reward_scores: list,
                      kl_values: list, loss: float,
                      prompts: list, completions: list):
        """Log GRPO training metrics per step."""
        if reward_scores:
            mean_r = sum(reward_scores) / len(reward_scores)
            std_r = (sum((r-mean_r)**2 for r in reward_scores)/len(reward_scores))**0.5
            self._w("reward/mean", mean_r, step)
            self._w("reward/std", std_r, step)
            self._w("reward/min", min(reward_scores), step)
            self._w("reward/max", max(reward_scores), step)

        if kl_values:
            mean_kl = sum(kl_values) / len(kl_values)
            self._w("kl/divergence_mean", mean_kl, step)
            self._w("kl/penalty_contribution", GRPO_KL_COEFF * mean_kl, step)

        self._w("policy/gradient_loss", loss, step)

        # Evasion/engagement rates from completions
        if completions:
            from src.training.reward_function import detect_evasive_response
            evasive = sum(1 for c in completions if detect_evasive_response(c))
            self._w("constitutional/evasion_rate", evasive / len(completions), step)
            engaged = sum(1 for c in completions if len(c.split()) > 30)
            self._w("constitutional/engagement_rate", engaged / len(completions), step)

    def log_alignment_eval(self, step: int, eval_results: dict):
        """Log before/after alignment metrics."""
        for stage in ["base", "sft", "grpo"]:
            if stage in eval_results:
                r = eval_results[stage]
                self._w(f"alignment/harmful_rate_{stage}", r.get("harmful_rate"), step)
                self._w(f"alignment/helpfulness_{stage}", r.get("helpfulness_score"), step)
                self._w(f"alignment/evasiveness_{stage}", r.get("evasiveness_rate"), step)

    def log_reward_distribution(self, step: int, scores: list):
        if self._ok and self.writer and scores:
            try:
                import torch
                self.writer.add_histogram("reward/distribution", torch.tensor(scores), step)
            except Exception:
                pass

    def log_hyperparams(self, config: dict):
        if self._ok and self.writer:
            try:
                self.writer.add_hparams(
                    {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
                    {"hparam/placeholder": 0})
            except Exception:
                pass

    def close(self):
        if self._ok and self.writer:
            self.writer.close()
