import traceback

try:
    from trl import SFTTrainer
    print("✅ TRL import OK")
except Exception as e:
    print(f"❌ TRL import failed:")
    traceback.print_exc()
