import json
import sys

with open('check.json', 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('reason') == 'compiler-message':
                msg = data['message']
                print(f"[{msg.get('level', 'info').upper()}] {msg['message']}")
                if msg.get('spans'):
                    for span in msg['spans']:
                        print(f"  at {span['file_name']}:{span['line_start']}:{span['column_start']}")
                if msg.get('rendered'):
                    print(msg['rendered'])
        except:
            pass
