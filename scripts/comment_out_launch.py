import os

file_path = "src/interface/python.rs"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Target lines are around 281-284 (0-indexed: 280-283)
# Check context
start_idx = 280
end_idx = 285

print("Context preview:")
for i in range(start_idx, end_idx):
    print(f"{i+1}: {lines[i].rstrip()}")

# Modify lines 281-284 to be commented out
# 281:             kernel.launch(
# 284:             ).map_err(...)?;

if "kernel.launch" in lines[280]:
    lines[280] = "            /* " + lines[280].lstrip()
    # Find closing )
    for i in range(281, 290):
        if ").map_err" in lines[i]:
            lines[i] = lines[i].rstrip() + " */\n"
            break
    print("Modified lines.")

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
