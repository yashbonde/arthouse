from functools import partial
import os

os.environ["NBOX_LOG_LEVEL"] = "info"
CACHE_NAME = os.environ.get("CACHE_NAME", "req_cache")
if not os.path.exists(CACHE_NAME):
  os.makedirs(CACHE_NAME)

import json
from random import choice
from fire import Fire
from uuid import uuid4
from tqdm import trange
from nbox import Operator
from base64 import b64decode
from PIL import Image

def raw(text: str, imgen = None):
  text = text.strip()
  if text == "":
    raise ValueError("Text cannot be empty")
  out = imgen.generate(text)
  with open(os.path.join(CACHE_NAME, "master.jsonl"), "a") as f:
    f.write(json.dumps({"id": str(uuid4()), "text": text, "keys": [out["id"]]}) + "\n")

def raw_submit(text, n: int = 4, imgen = None, tags = []):
  text = text.strip()
  if text == "":
    raise ValueError("Text cannot be empty")
  out = imgen.generate_lp(text, batch_size = n)
  with open(os.path.join(CACHE_NAME, "master.jsonl"), "a") as f:
    f.write(json.dumps({"id": str(uuid4()), "text": text, "keys": out["id"], "tags": tags}) + "\n")

def raw_submit_json(json_file: str, imgen = None):
  with open(json_file, "r") as f:
    data = json.load(f)
    for x in data:
      raw_submit(x, imgen)

def submit(text: str, style = None, n = 4, imgen = None):
  text = text.strip()
  if text == "":
    raise ValueError("Text cannot be empty")
  with open("./styles/styles.json", "r") as f:
    styles = json.load(f)
  
  if style is None:
    style = choice(list(styles.keys()))
  if style not in styles:
    raise ValueError(f"Style {style} not found")
  style_prompts = styles[style]
  n_prompt = n // len(style_prompts)
  for t in style_prompts:
    text += " " + t
    raw_submit(text, n_prompt, imgen)

def get(key: str, imgen = None):
  out = imgen.get(key)
  if not out["success"]:
    print(out["message"])
    return
  img = Image.frombytes("RGB", data = b64decode(out["image"]), size = (512, 512))
  img.save(f"images/{key}.png")
  print(f"Saved image to images/{key}.png")
  return

def download(cache: str, imgen = None):
  keys = []
  with open("req_cache/master.jsonl", "r") as f:
    for line in f:
      data = json.loads(line)
      if data["id"] != cache:
        continue
      keys = data["keys"]

  if not keys:
    print("Cache not found")
    return

  keys_obtained = []
  pbar = trange(len(keys))
  skipped = 0
  not_avail = 0
  downloaded = 0
  for i in pbar:
    pbar.set_description(f"{skipped} skipped | {not_avail} NA | {downloaded} downloaded")
    k = keys[i]
    os.makedirs(f"images/{cache}", exist_ok=True)
    if not os.path.exists(f"images/{cache}/{k}.png"):
      out = imgen.get(k)
      # print(out)
      if not out["success"]:
        not_avail += 1
      else:
        img = Image.frombytes("RGB", data = b64decode(out["image"]), size = (512, 512))
        img.save(f"images/{cache}/{k}.png")
        downloaded += 1
        keys_obtained.append(k)
    else:
      skipped +=1
      keys_obtained.append(k)

  keys_obtained = ["images/" + x + ".png" for x in keys_obtained]

def _status(stat, n_gpu = 1):
  est_time_min = stat * 30 / 60 / 60 / n_gpu
  est_time_max = stat * 35 / 60 / 60 / n_gpu
  # unit = "hours"

  hours_min = int(est_time_min)
  minutes_min = int((est_time_min*60) % 60)
  seconds_min = int((est_time_min*3600) % 60)
  min_str = f"{hours_min}:{minutes_min:02d}:{seconds_min:02d} hrs"

  hours_max = int(est_time_max)
  minutes_max = int((est_time_max*60) % 60)
  seconds_max = int((est_time_max*3600) % 60)
  max_str = f"{hours_max}:{minutes_max:02d}:{seconds_max:02d} hrs"

  print(f"on {n_gpu} GPUs")
  print(f"Total pending itmes: {stat}")
  print(f"Estimated time: {min_str} - {max_str}")

def status(imgen):
  data = imgen.status()
  # n_gpu = data["n_gpu"]
  stat = data["qsize"]
  _status(stat, 2)

def open_key(key):
  img = Image.open(f"images/{key}.png")
  img.show()
  return

def get_client():
  imgen = Operator.from_serving(
    url = "https://stablediff-amruk.build.nimblebox.ai/",
    token = ""
  )
  return imgen

if __name__ == "__main__":
  imgen = get_client()
  
  # print(imgen._op_spec)
  # print(imgen._op_type)

  Fire({
    "raw": partial(raw, imgen = imgen),
    "raw_submit": partial(raw_submit, imgen = imgen),
    "raw_json": partial(raw_submit_json, imgen = imgen),
    "submit": partial(submit, imgen = imgen),
    "get": partial(get, imgen = imgen),
    "download": partial(download, imgen = imgen),
    "status": partial(status, imgen = imgen),
    "open": open_key
  })

