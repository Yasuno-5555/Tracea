
import sys
import os

try:
    with open("doctor_log.txt", "r", encoding="utf-16") as f:
        content = f.read()
except:
    try:
        with open("doctor_log.txt", "r", encoding="utf-8") as f:
            content = f.read()
    except:
        with open("doctor_log.txt", "rb") as f:
            content = f.read().decode("utf-16-le", errors="ignore")

lines = content.splitlines()
show = False
for line in lines:
    if "[RuntimeManager] ptxas FAILED" in line:
        show = True
    if show:
        print(line)
    if "Attention Compilation Failed" in line:
        show = False
    
    # Also capture the command itself
    if "Running ptxas for" in line:
        print(line)
