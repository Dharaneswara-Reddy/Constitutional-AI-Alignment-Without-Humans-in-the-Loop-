#!/usr/bin/env python
"""Quick setup script to add HF_TOKEN to .env file."""

import os
from pathlib import Path

print("=" * 70)
print("HUGGING FACE TOKEN SETUP")
print("=" * 70)
print()
print("This will add your HF token to .env for faster downloads.")
print()
print("Get your token at: https://huggingface.co/settings/tokens")
print()

token = input("Paste your HF token (starts with hf_): ").strip()

if not token:
    print("\n❌ No token provided. Skipping.")
    exit(0)

if not token.startswith("hf_"):
    print("\n⚠️  Warning: Token should start with 'hf_'")
    confirm = input("Continue anyway? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        exit(0)

# Read .env file
env_path = Path(".env")
if not env_path.exists():
    print("\n❌ .env file not found!")
    exit(1)

with open(env_path, 'r') as f:
    lines = f.readlines()

# Update or add HF_TOKEN
updated = False
new_lines = []
for line in lines:
    if line.startswith("HF_TOKEN=") or line.startswith("# HF_TOKEN="):
        new_lines.append(f"HF_TOKEN={token}\n")
        updated = True
    else:
        new_lines.append(line)

if not updated:
    # Add at the end
    new_lines.append(f"\n# Hugging Face Token\nHF_TOKEN={token}\n")

# Write back
with open(env_path, 'w') as f:
    f.writelines(new_lines)

print("\n✅ HF_TOKEN added to .env file!")
print()
print("Next steps:")
print("  1. Restart your Jupyter kernel (Kernel → Restart Kernel)")
print("  2. Run Cell 2 to reload environment variables")
print("  3. Continue training - downloads will be faster!")
print()
print("=" * 70)
