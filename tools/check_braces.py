
path = "src/emitter/cuda.rs"
with open(path, "r") as f:
    text = f.read()

count = 0
line_num = 1
for i, char in enumerate(text):
    if char == '\n':
        line_num += 1
    if char == '{':
        count += 1
    elif char == '}':
        count -= 1
        if count < 0:
            print(f"Negative balance at line {line_num}")

print(f"Final Count: {count}")
