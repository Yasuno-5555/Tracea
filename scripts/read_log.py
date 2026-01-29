
import sys

try:
    with open("verification_debug.txt", "r", encoding="utf-16-le", errors='replace') as f:
        print(f.read())
except Exception:
    try:
        with open("verification_debug.txt", "r", encoding="utf-8", errors='replace') as f:
            print(f.read())
    except Exception as e:
        print(f"Failed to read log: {e}")
