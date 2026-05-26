#!/usr/bin/env python3
"""
Emergency fix for trailing whitespace corruption in JSON keys and string values.
Observed in bot_settings.json and hypothesis_db.json.
Run ONCE before restarting dotm_sniper.py and hermes_advisor.py.
"""
import json
import os
import shutil
from datetime import datetime
from typing import Any

def normalize_keys_and_values(obj: Any) -> Any:
    """Recursively strip whitespace from dict keys AND string values."""
    if isinstance(obj, dict):
        return {
            (k.strip() if isinstance(k, str) else k): normalize_keys_and_values(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [normalize_keys_and_values(item) for item in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj

def sanitize_file(path: str) -> bool:
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, skipping")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.backup_{timestamp}"
    shutil.copy2(path, backup_path)
    print(f"✅ Backup created: {backup_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    normalized = normalize_keys_and_values(data)

    temp_path = f"{path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    os.replace(temp_path, path)
    print(f"✅ Sanitized: {path}")
    return True

if __name__ == "__main__":
    import sys
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "/root/dotm-sniper/bot_settings.json",
        "/root/dotm-sniper/hypothesis_db.json",
    ]
    print("=" * 60)
    print("  JSON Key/Value Sanitizer — Emergency Fix")
    print("=" * 60)
    for path in files:
        sanitize_file(path)
    print("\n✅ Done. Restart bot and hermes after applying code patches.")