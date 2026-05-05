"""
Dataset Preparation — formats HH-RLHF for SFT and GRPO pipelines.
Paper: Bai et al. 2022, Sections 3.1 (critique+revision) and 3.2 (helpfulness).

For SFT: apply critique+revision on REJECTED responses (harmful ones).
         Mix in CHOSEN responses as helpfulness samples (Section 3.2).
For GRPO: extract prompts only — mix 40% red-team, 40% ambiguous, 20% normal.
"""

import json, os, random, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (DATASET_DIR, SFT_DATA_PATH, GRPO_PROMPT_PATH,
                    SFT_MAX_SAMPLES, GRPO_MAX_PROMPTS, GROQ_API_KEY,
                    GROQ_MODEL, GROQ_TEMPERATURE, API_DELAY_SECONDS)

_CONST_PATH = Path(__file__).parent.parent.parent / "constitution.json"
with open(_CONST_PATH) as f:
    _CONST = json.load(f)
SL_PRINCIPLES = _CONST["sl_principles"]


def parse_hh_conversation(text: str) -> dict:
    """Parse HH-RLHF conversation string into prompt + final response."""
    parts = text.split("\n\nAssistant:")
    if len(parts) < 2:
        return {"prompt": text.strip(), "response": "", "full_text": text}
    response = parts[-1].strip()
    prompt_part = "\n\nAssistant:".join(parts[:-1])
    human_parts = prompt_part.split("\n\nHuman:")
    last_human = human_parts[-1].strip() if len(human_parts) > 1 else prompt_part.strip()
    return {"prompt": last_human, "response": response, "full_text": text}


def apply_critique_revision_chain(prompt: str, bad_response: str, principle: dict) -> dict:
    """
    3-step critique+revision chain (paper Section 3.1).
    Step A: the bad_response is already provided (from rejected HH-RLHF entry).
    Step B: critique it using principle's critique_request.
    Step C: revise using principle's revision_request.
    Uses Groq API as critic/reviser.
    """
    from groq import Groq
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Export it first.")
    client = Groq(api_key=GROQ_API_KEY)

    # Step B — Critique
    critique_msgs = [
        {"role": "system", "content": "You are a critical AI safety evaluator. Be concise."},
        {"role": "user", "content":
            f"Human request: {prompt}\n\nAI response: {bad_response}\n\n"
            f"{principle['critique_request']}"}
    ]
    time.sleep(API_DELAY_SECONDS)
    c_resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=critique_msgs, max_tokens=300,
        temperature=GROQ_TEMPERATURE)
    critique = c_resp.choices[0].message.content.strip()

    # Step C — Revision
    revision_msgs = [
        {"role": "system", "content": "You are a helpful, harmless AI assistant."},
        {"role": "user", "content":
            f"Human request: {prompt}\n\nOriginal AI response: {bad_response}\n\n"
            f"Critique: {critique}\n\n{principle['revision_request']}"}
    ]
    time.sleep(API_DELAY_SECONDS)
    r_resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=revision_msgs, max_tokens=400,
        temperature=GROQ_TEMPERATURE)
    revision = r_resp.choices[0].message.content.strip()

    return {"prompt": prompt, "revision": revision, "critique": critique,
            "principle_id": principle["id"]}


def load_hh_rlhf(split: str) -> list:
    path = os.path.join(DATASET_DIR, f"hh_rlhf_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run download_datasets.py first. Missing: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prepare_sft_dataset(max_samples: int = SFT_MAX_SAMPLES):
    """
    Creates SFT dataset from HH-RLHF using critique+revision chain.
    Uses REJECTED responses as starting point (paper Section 3.1).
    Mixes in CHOSEN responses as helpfulness samples (paper Section 3.2).
    Saves to datasets/sft_dataset.jsonl.
    """
    print("=" * 60)
    print("Preparing SFT Dataset")
    print(f"  Max samples: {max_samples}")
    print("=" * 60)
    os.makedirs(os.path.dirname(SFT_DATA_PATH), exist_ok=True)
    rows = load_hh_rlhf("train")
    random.shuffle(rows)

    # Split budget: 60% harmlessness (critique+revision), 40% helpfulness (direct)
    n_harmless = int(max_samples * 0.6)
    n_helpful = max_samples - n_harmless
    results = []

    # --- Harmlessness samples: critique+revision on REJECTED responses ---
    print(f"\n[sft] Processing {n_harmless} harmlessness samples (critique+revision)...")
    for i, row in enumerate(rows[:n_harmless]):
        try:
            rejected = parse_hh_conversation(row.get("rejected", ""))
            if not rejected["prompt"] or not rejected["response"]:
                continue
            principle = random.choice(SL_PRINCIPLES)
            result = apply_critique_revision_chain(
                rejected["prompt"], rejected["response"], principle)
            result["type"] = "harmless"
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_harmless}] critique+revision done")
                # Save incrementally
                with open(SFT_DATA_PATH, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
        except Exception as e:
            print(f"  [sft] Error at sample {i}: {e}")
            continue

    # --- Helpfulness samples: use CHOSEN responses directly (Section 3.2) ---
    print(f"\n[sft] Adding {n_helpful} helpfulness samples (chosen responses)...")
    helpful_rows = rows[n_harmless:n_harmless + n_helpful * 2]
    added = 0
    for row in helpful_rows:
        if added >= n_helpful:
            break
        try:
            chosen = parse_hh_conversation(row.get("chosen", ""))
            if not chosen["prompt"] or not chosen["response"]:
                continue
            results.append({
                "prompt": chosen["prompt"],
                "revision": chosen["response"],
                "critique": "",
                "principle_id": 0,
                "type": "helpful"
            })
            added += 1
        except Exception:
            continue

    with open(SFT_DATA_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n[sft] Dataset saved: {len(results)} samples → {SFT_DATA_PATH}")
    harmless_n = sum(1 for r in results if r["type"] == "harmless")
    helpful_n = sum(1 for r in results if r["type"] == "helpful")
    print(f"  Harmlessness: {harmless_n} | Helpfulness: {helpful_n}")
    return results


def prepare_grpo_prompts(max_prompts: int = GRPO_MAX_PROMPTS):
    """
    Extracts prompts from HH-RLHF for GRPO online training.
    Mix: 40% red-team (rejected), 40% ambiguous (chosen), 20% normal (benign).
    GRPO generates its own responses — only prompts needed.
    Saves to datasets/grpo_prompts.jsonl.
    """
    print("\n" + "=" * 60)
    print(f"Preparing GRPO Prompt Dataset ({max_prompts} prompts)")
    print("=" * 60)
    os.makedirs(os.path.dirname(GRPO_PROMPT_PATH), exist_ok=True)
    rows = load_hh_rlhf("train")
    random.shuffle(rows)

    n_red = int(max_prompts * 0.4)
    n_amb = int(max_prompts * 0.4)
    n_normal = max_prompts - n_red - n_amb

    prompts = []
    seen = set()

    for row in rows:
        if len(prompts) >= max_prompts:
            break
        cat_idx = len(prompts)

        if cat_idx < n_red:
            # Red-team: prompts from REJECTED conversations (harmful intent)
            parsed = parse_hh_conversation(row.get("rejected", ""))
            category = "red_team"
        elif cat_idx < n_red + n_amb:
            # Ambiguous: prompts from CHOSEN (could be answered safely or not)
            parsed = parse_hh_conversation(row.get("chosen", ""))
            category = "ambiguous"
        else:
            # Normal: benign helpfulness prompts
            parsed = parse_hh_conversation(row.get("chosen", ""))
            category = "normal"

        p = parsed["prompt"].strip()
        if not p or len(p) < 10 or p in seen:
            continue
        seen.add(p)
        prompts.append({"prompt": p, "category": category})

    # IMPORTANT — API Cost Warning (per spec):
    # GRPO samples G=4 completions per prompt per training step.
    # With {max_prompts} prompts and 1 epoch = ~{max_prompts * 4} Groq API calls.
    # At Groq free tier (14,400 req/day on llama-3.3-70b), budget accordingly.

    with open(GRPO_PROMPT_PATH, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    counts = {c: sum(1 for p in prompts if p["category"] == c)
              for c in ["red_team", "ambiguous", "normal"]}
    print(f"\n[grpo] Prompt dataset ready:")
    print(f"  Red-team: {counts['red_team']} | Ambiguous: {counts['ambiguous']} | Normal: {counts['normal']}")
    print(f"  Total: {len(prompts)} → {GRPO_PROMPT_PATH}")
    print(f"  Est. API calls during GRPO: ~{len(prompts) * 4} (G=4 per prompt)")
    return prompts


def get_dataset_summary() -> dict:
    """Returns stats dict for Streamlit dashboard display."""
    summary = {}
    for split in ["train", "test"]:
        path = os.path.join(DATASET_DIR, f"hh_rlhf_{split}.jsonl")
        if os.path.exists(path):
            rows = sum(1 for _ in open(path) if _.strip())
            summary[split] = {"rows": rows, "size_mb": round(os.path.getsize(path)/1e6, 1)}
        else:
            summary[split] = {"rows": 0, "size_mb": 0}
    for key, path in [("sft", SFT_DATA_PATH), ("grpo", GRPO_PROMPT_PATH)]:
        if os.path.exists(path):
            rows = sum(1 for _ in open(path) if _.strip())
            summary[key] = {"rows": rows, "size_mb": round(os.path.getsize(path)/1e6, 1)}
        else:
            summary[key] = {"rows": 0, "size_mb": 0}
    return summary


if __name__ == "__main__":
    prepare_sft_dataset()
    prepare_grpo_prompts()
