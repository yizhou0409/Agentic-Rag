import json
from tqdm import tqdm

with open("wiki-20181220-chunk100w-old.jsonl", "r") as f:
    data = []
    lines = f.readlines()
    for line in lines:
        data.append(json.loads(line))
    print(f"Number of lines: {len(data)}")

new_data = []
for item in tqdm(data):
    pid = item["id"]
    content = item["contents"]
    title = content.split("\n")[0].strip()
    text = content.split("\n")[1].strip()

    new_data.append({
        "id": pid,
        "title": title,
        "text": text
    })

with open("wiki-20181220-chunk100w.jsonl", "w") as f:
    for item in new_data:
        f.write(json.dumps(item) + "\n")