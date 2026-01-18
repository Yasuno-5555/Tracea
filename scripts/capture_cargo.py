import subprocess
import os

print("Running cargo check...")
res = subprocess.run(["cargo", "check", "--message-format=short"], capture_output=True, text=True, encoding="utf-8")

with open("captured_errors.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(res.stdout)
    f.write("\nSTDERR:\n")
    f.write(res.stderr)

print("Done. Errors written to captured_errors.txt")
