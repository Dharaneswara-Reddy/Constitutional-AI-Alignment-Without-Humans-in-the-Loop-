"""
Constitutional Reward Function — KL-Regularized GRPO.
Paper: Bai et al. 2022, Sections 4.1, 4.3, 4.4.

Replaces paper's pre-trained LM feedback model with llama-3.3-70b via Groq API.

Key paper methods implemented:
  - Principle ensembling (Section 4.3): sample 4 of 16 RL principles, average scores
  - Soft labels clamped to [0.4, 0.6] (Section 4.3): not hard 0/1
  - Evasiveness penalty (Section 4.4): penalize flat refusals
  - Engagement bonus: reward responses that explain rather than refuse

TRL GRPOTrainer adds -beta*KL(pi_theta||pi_ref) automatically when beta= is set in GRPOConfig.
"""

import json, os, random, time, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, API_DELAY_SECONDS,
                    GRPO_NUM_GENERATIONS, REWARD_SAFE, REWARD_PARTIAL, REWARD_HARMFUL,
                    REWARD_CLAMP_LOW, REWARD_CLAMP_HIGH, REWARD_ENSEMBLE_N,
                    REWARD_LOG_PATH, LOG_DIR)

# Load constitution
_CONST_PATH = Path(__file__).parent.parent.parent / "constitution.json"
with open(_CONST_PATH) as f:
    _CONST = json.load(f)
RL_PRINCIPLES = _CONST["rl_principles"]

os.makedirs(LOG_DIR, exist_ok=True)


def _get_groq_client():
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")
    return Groq(api_key=GROQ_API_KEY)


def detect_evasive_response(response: str) -> bool:
    """
    Detects boilerplate evasive responses (paper Section 4.4).
    These are safe but unhelpful — paper explicitly penalizes them.
    """
    if not response or len(response.strip().split()) < 3:
        return True
    r = response.lower().strip()
    evasive_patterns = [
        r"^i (can'?t|cannot|won'?t|will not|am unable)",
        r"^i (apologize|am sorry|do not)",
        r"^i'?m (unable|not able|sorry)",
        r"^(sorry,? )?i (don'?t|cannot|can'?t)",
        r"^as an ai,? i",
        r"^i must (decline|refuse)",
        r"^(i'?m afraid|unfortunately,?) i",
    ]
    for pat in evasive_patterns:
        if re.match(pat, r):
            # Only evasive if short (under 30 words) — long responses that start
            # with a refusal but then explain are NOT evasive (paper Section 4.4)
            if len(response.split()) < 30:
                return True
    return False


def get_groq_constitutional_score(prompt: str, response: str,
                                   principle: str = None) -> float:
    """
    Single-principle scoring via Groq API.
    Paper Section 4.1: present response pair, compute relative probability.

    Uses a baseline "evasive response" as Response A and the actual response
    as Response B. Asks which better follows the constitutional principle.
    Soft label clamped to [REWARD_CLAMP_LOW, REWARD_CLAMP_HIGH] (Section 4.3).
    Returns 0.0 (unclamped) for clearly harmful responses.
    """
    if principle is None:
        principle = random.choice(RL_PRINCIPLES)

    # Baseline: clearly evasive response (known-bad)
    baseline = "I cannot help with that request."

    system = "You are a constitutional AI judge. Score AI responses for safety and helpfulness."
    user = f"""Evaluate two AI responses to the same prompt according to this principle:

PRINCIPLE: {principle}

PROMPT: {prompt}

RESPONSE A: {baseline}

RESPONSE B: {response[:600]}

Which response better follows the principle? Reply with ONLY a JSON object:
{{"winner": "A" or "B", "score_b": 0.0-1.0, "reason": "one sentence"}}

score_b is the constitutional compliance score for Response B (0=bad, 1=good)."""

    try:
        client = _get_groq_client()
        time.sleep(API_DELAY_SECONDS)
        resp = client.chat.completions.create(
            model=GROQ_MODEL, max_tokens=100, temperature=GROQ_TEMPERATURE,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        raw = resp.choices[0].message.content.strip()
        # Extract JSON
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return REWARD_PARTIAL
        result = json.loads(match.group())
        score = float(result.get("score_b", REWARD_PARTIAL))
        # Clamp to [0.4, 0.6] per paper Section 4.3 (prevents overconfident labels)
        # But return 0.0 for clearly harmful (unclamped)
        if score < 0.2:
            return REWARD_HARMFUL  # Clearly harmful — no clamping
        return max(REWARD_CLAMP_LOW, min(REWARD_CLAMP_HIGH, score))
    except Exception as e:
        print(f"  [reward] Groq error: {e} — returning neutral {REWARD_PARTIAL}")
        return REWARD_PARTIAL


def compute_ensembled_reward(prompt: str, response: str,
                              n_principles: int = REWARD_ENSEMBLE_N) -> float:
    """
    Ensemble over multiple RL principles (paper Section 4.3).
    "Ensembling over principles led to notably more robust PM behavior."
    Samples n_principles randomly from all 16 RL principles, averages scores.
    """
    selected = random.sample(RL_PRINCIPLES, min(n_principles, len(RL_PRINCIPLES)))
    scores = []
    for p in selected:
        s = get_groq_constitutional_score(prompt, response, p)
        scores.append(s)
    return sum(scores) / len(scores)


def log_reward_to_file(prompt: str, completion: str, score: float):
    """Append to logs/reward_log.jsonl for TensorBoard and dashboard charts."""
    entry = {
        "prompt": prompt[:100],
        "completion": completion[:150],
        "score": score,
        "timestamp": time.time()
    }
    try:
        with open(REWARD_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def constitutional_reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    TRL GRPOTrainer reward function — required signature: (prompts, completions, **kwargs).

    For each completion:
      1. Detect purely evasive response → score = 0.0 (paper Section 4.4)
      2. Compute ensembled constitutional score (4 principles, clamped [0.4,0.6])
      3. Engagement bonus +0.1 for responses >30 words that explain (not just refuse)
      4. Log to reward_log.jsonl

    TRL automatically adds -beta*KL(pi_theta||pi_ref) when beta= is set in GRPOConfig.
    """
    scores = []
    # Expand prompts: each prompt was used for GRPO_NUM_GENERATIONS completions
    expanded = [p for p in prompts for _ in range(GRPO_NUM_GENERATIONS)]
    n = len(completions)
    print(f"\n  [reward_fn] Scoring {n} completions ({len(prompts)} prompts x G={GRPO_NUM_GENERATIONS})")

    for i, (prompt, completion) in enumerate(zip(expanded, completions)):
        print(f"  [{i+1}/{n}] prompt[:40]: {prompt[:40]!r}...")
        if detect_evasive_response(completion):
            score = REWARD_HARMFUL  # 0.0 — evasion penalty (paper Section 4.4)
            print(f"    → EVASIVE (score=0.0)")
        else:
            score = compute_ensembled_reward(prompt, completion)
            # Engagement bonus: longer, explanatory responses beat flat safe ones
            if len(completion.split()) > 30:
                score = min(REWARD_SAFE, score + 0.1)
            print(f"    → score={score:.3f} | len={len(completion.split())}w")
        scores.append(score)
        log_reward_to_file(prompt, completion, score)

    mean_r = sum(scores) / len(scores) if scores else 0
    print(f"  [reward_fn] Batch mean={mean_r:.3f} | min={min(scores):.3f} | max={max(scores):.3f}")
    return scores
