"""
Dataset Download Pipeline — HuggingFace HH-RLHF via curl API.
Paper: Bai et al. 2022 uses Anthropic/hh-rlhf dataset.

Two strategies:
  A — Parquet bulk download (preferred, fast)
  B — Row API fallback (if parquet fails)
"""

import subprocess, json, os, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATASET_DIR


def get_parquet_urls(split: str) -> list:
    cmd = (f'curl -sX GET "https://huggingface.co/api/datasets/Anthropic/hh-rlhf'
           f'/parquet/default/{split}" -H "Accept: application/json"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    data = json.loads(r.stdout)
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Unexpected response: {data}")


def download_via_parquet(split: str) -> str:
    import pandas as pd
    print(f"\n[dl] Fetching parquet URLs for split='{split}'...")
    infos = get_parquet_urls(split)
    print(f"[dl] Found {len(infos)} parquet file(s)")
    parquet_files = []
    for i, info in enumerate(infos):
        url = info["url"]
        out = os.path.join(DATASET_DIR, f"hh_rlhf_{split}_{i}.parquet")
        print(f"[dl] Downloading part {i+1}/{len(infos)}...")
        subprocess.run(f'curl -L "{url}" -o "{out}" --progress-bar 2>&1', shell=True)
        parquet_files.append(out)
    merged = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    out_jsonl = os.path.join(DATASET_DIR, f"hh_rlhf_{split}.jsonl")
    merged.to_json(out_jsonl, orient="records", lines=True)
    print(f"[dl] Saved {len(merged):,} rows → {out_jsonl}")
    for f in parquet_files:
        try: os.remove(f)
        except OSError: pass
    return out_jsonl


def download_via_rows_api(split: str, max_rows: int = 5000) -> str:
    out_jsonl = os.path.join(DATASET_DIR, f"hh_rlhf_{split}.jsonl")
    all_rows, offset = [], 0
    print(f"\n[dl] Rows API fallback: '{split}' (max {max_rows})...")
    while offset < max_rows:
        url = (f"https://datasets-server.huggingface.co/rows"
               f"?dataset=Anthropic%2Fhh-rlhf&config=default&split={split}"
               f"&offset={offset}&length=100")
        r = subprocess.run(f'curl -sX GET "{url}" -H "Accept: application/json"',
                           shell=True, capture_output=True, text=True)
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            break
        rows = [x["row"] for x in data.get("rows", [])]
        if not rows: break
        all_rows.extend(rows)
        offset += len(rows)
        print(f"  {len(all_rows):,} rows downloaded...")
        time.sleep(0.2)
    with open(out_jsonl, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[dl] Saved {len(all_rows):,} rows → {out_jsonl}")
    return out_jsonl


def download_hh_rlhf_splits():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("=" * 60)
    print("HH-RLHF Dataset Download — Anthropic/hh-rlhf")
    print("=" * 60)
    for split, fallback_max in [("train", 10000), ("test", 2000)]:
        out = os.path.join(DATASET_DIR, f"hh_rlhf_{split}.jsonl")
        if os.path.exists(out):
            print(f"[dl] '{split}' exists ({os.path.getsize(out)/1e6:.1f} MB), skipping")
            continue
        try:
            download_via_parquet(split)
        except Exception as e:
            print(f"[dl] Parquet failed ({e}), using rows API...")
            download_via_rows_api(split, max_rows=fallback_max)


def print_dataset_stats():
    print("\n" + "=" * 60 + "\nDataset Statistics\n" + "=" * 60)
    for split in ["train", "test"]:
        path = os.path.join(DATASET_DIR, f"hh_rlhf_{split}.jsonl")
        if not os.path.exists(path):
            print(f"  {split}: NOT FOUND")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        avg_c = sum(len(r.get("chosen","").split()) for r in rows) / max(len(rows),1)
        avg_r = sum(len(r.get("rejected","").split()) for r in rows) / max(len(rows),1)
        print(f"\n  {split.upper()}: {len(rows):,} rows | avg chosen {avg_c:.0f}w | avg rejected {avg_r:.0f}w")
        if rows:
            print(f"  Sample chosen[:100]: {rows[0].get('chosen','')[:100].replace(chr(10),' ')}")
    print("=" * 60)


if __name__ == "__main__":
    download_hh_rlhf_splits()
    print_dataset_stats()
