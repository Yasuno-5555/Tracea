
import json
import sys

count = 0
try:
    with open("check_output.json", "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                # Some lines might be mixed text/json if stdout/stderr mixed. 
                # Attempt to find '{'
                start = line.find('{')
                if start != -1:
                    line = line[start:]
                
                msg = json.loads(line)
                if "message" in msg and isinstance(msg["message"], dict):
                    level = msg["message"].get("level")
                    if level == "error":
                        print("-" * 40)
                        print(msg["message"].get("rendered", "No rendered message"))
                        print("-" * 40)
                        count += 1
            except:
                pass
except Exception as e:
    print(f"Error reading file: {e}")

if count == 0:
    print("No error messages found in JSON.")
