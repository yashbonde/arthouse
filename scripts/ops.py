from fire import Fire
from uuid import uuid4
from subprocess import Popen

import json

def merge(ow: bool = True):
  l = 0
  all_data = {}
  with open("./req_cache/master.jsonl", "r") as f:
    for line in f:
      if not line:
        continue
      data = json.loads(line)
      all_data.setdefault(data["text"], []).extend(data["keys"])
      l += 1
  print("before:", l)
  print("after:", len(all_data))

  with open("./req_cache/master_merged.jsonl", "w") as f:
    for text, keys in all_data.items():
      data = {"id": str(uuid4()), "text": text, "keys": list(set(keys))}
      f.write(json.dumps(data) + "\n")

  if ow:
    Popen(["cp", "./req_cache/master_merged.jsonl", "./req_cache/master.jsonl"])


if __name__ == "__main__":
  Fire({
    "merge": merge,
  })
