
import sys
print("Debug Script Starting", file=sys.stderr)
sys.stdout.flush()
try:
    import torch
    print("Torch Imported", file=sys.stderr)
except ImportError:
    print("Torch Missing", file=sys.stderr)

import os
release_path = os.path.join(os.getcwd(), "target", "release")
sys.path.insert(0, release_path)
print(f"Path inserted: {release_path}", file=sys.stderr)

try:
    import tracea
    print("Tracea Imported", file=sys.stderr)
    ctx = tracea.Context("sm_86")
    print("Context Created", file=sys.stderr)
    try:
        diagnosis = ctx.doctor.diagnose()
        print(f"Diagnosis: {diagnosis.status}", file=sys.stderr)
    except Exception as e:
        print(f"Diagnosis Failed: {e}", file=sys.stderr)
except Exception as e:
    print(f"Tracea Failed: {e}", file=sys.stderr)
print("Debug Script Done", file=sys.stderr)
