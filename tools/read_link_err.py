
try:
    with open("link_err.txt", "rb") as f:
        data = f.read()
        text = data.decode('utf-8', errors='ignore')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        print(text)
except Exception as e:
    print(e)
