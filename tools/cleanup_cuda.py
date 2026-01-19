
path = "src/emitter/cuda.rs"
with open(path, "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    # Line 596 in 1-based index is i=595.
    if i >= 595 and "format!(r#\"" in line:
        skip = True
    
    if skip:
        if "\"#)" in line:
            skip = False
        continue
    
    new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)

print(f"Cleaned {path}")
