
try:
    with open("build_err_clean.txt", "rb") as f:
        data = f.read()
        text = data.decode('utf-8', errors='ignore')
        # Replace \r\n with \n, then \r with \n to handle progress bar overwrites
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        print(text)
except Exception as e:
    print(e)
