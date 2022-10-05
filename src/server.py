import plyvel
import random
from time import sleep
from PIL import Image
from io import BytesIO
from typing import List
from requests import Session
from threading import Thread
from time import sleep, time_ns
from queue import PriorityQueue, Full, Empty

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse

from diffusers import StableDiffusionPipeline
from torch import autocast
import torch

from nbox import operator, Operator
from nbox.utils import b64encode


class DataItem:
  def __init__(self, p, **kwargs):
    self.p = p
    self.__dict__.update(kwargs)

  def __lt__(self, other: 'DataItem'):
    # required for priority queue
    return self.p < other.p


def processor(leveldb, queue, device = "cuda:0"):
  print("loading model on:", device)
  pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token = "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",
    cache_dir = "./",
  ).to(device)

  while True:
    try:
      item = queue.get()
    except Empty:
      sleep(0.1)

    if item is None:
      break

    _, item = item
    with torch.no_grad():
      with autocast("cuda" if device.startswith("cuda") else "cpu"):
        out = pipe(
          prompt = item.text,
          height = item.height,
          width = item.width,
          num_inference_steps = item.num_inference_steps,
          guidance_scale = item.guidance_scale,
          eta = item.eta
        )
        if out["nsfw_content_detected"]:
          # prevent blanks
          continue
        image = out["sample"][0]
        leveldb.put(item.resp_id.encode(), image.tobytes())


@operator()
class GenImage:
  def __init__(self):
    max_queue_size = 4096
    self.queue = PriorityQueue(max_queue_size)
    self.leveldb = plyvel.DB("ldb_raw", create_if_missing = True, error_if_exists = False)

    # start the processor thread
    self.processors = []
    for i in range(torch.cuda.device_count()):
      p = Thread(target=processor, args=(self.leveldb, self.queue, f"cuda:{i}"))
      p.start()
      self.processors.append(p)
      sleep(60) # sleep 60 seconds before loading another model

  def _generate(
    self,
    priority: int,
    resp_id: str,
    text: str,
    n_iter: int,
    height = 512,
    width = 512,
    num_inference_steps = 100,
    guidance_scale = 20,
    eta = 0.4
  ):
    self.queue.put((
      priority,
      DataItem(
        p = priority,
        resp_id = resp_id,
        text = text,
        n_iter = int(n_iter),
        height = int(height),
        width = int(width),
        num_inference_steps = int(num_inference_steps),
        guidance_scale = float(guidance_scale),
        eta = float(eta)
      )
    ))

  def generate(self, text: str, n_iter: int = 100, guidance_scale = 20, eta = 0.4) -> str:
    """This endpoint is used to generate a single image, these queries have highest priority"""
    _id = "key-" + str(time_ns())
    try:
      self._generate(0, resp_id = _id, text = text, n_iter = n_iter, guidance_scale = guidance_scale, eta=eta)
    except Full:
      return {"success": False, "message": "Queue is full"}
    return {"success": True, "id": _id}

  def get(self, id: str):
    out = self.leveldb.get(id.encode())
    if out is None:
      return {"success": False, "message": "Image not found"}
    return {
      "success": True,
      "image": b64encode(out),
    }

  def get_random(self, offset: int = 100):
    """Get a random item from levelDB"""
    offset = int(offset)
    for i, (k, v) in enumerate(self.leveldb.iterator()):
      if random.randint(0, i) > offset:
        continue
      return {
        "success": True,
        "image": b64encode(k),
      }
    return {"success": False, "message": "No images found"}

  def status(self):
    # tell about the status of the service like pending jobs, etc.
    return {"qsize": self.queue.qsize(), "n_gpu": torch.cuda.device_count()}

  def generate_lp(self, text: str, n_iter: int = 100, batch_size: int = 1, guidance_scale = 20, eta = 0.4) -> List[str]:
    """This endpoint is used to generate a bulk of images, these queries have lowest priority"""
    ids = []
    for _ in range(batch_size):
      _id = "key-" + str(time_ns())
      try:
        self._generate(1, resp_id = _id, text = text, n_iter = n_iter, guidance_scale = guidance_scale, eta=eta)
      except Full:
        return {"success": False, "message": "Queue is full, please try later"}
      ids.append(_id)
    return {"success": True, "id": ids}

def get_client():
  img_gen = Operator.from_serving(
    url = "https://stablediff-amruk.build.nimblebox.ai/",
    token = "Y7jD8cgzd41PmSQ"
  )
  return img_gen


if __name__ == "__main__":
  genimage = GenImage()
  # for i in range(6):
  #   if i % 2 == 0:
  #     id = genimage.generate("a lady sitting in office eating noodles for lunch during noon. pencil sketch.", 10)
  #     print(i, id)
  #   else:
  #     ids = genimage.generate_lp("a lady sitting in office eating noodles for lunch during noon. pencil sketch.", 10, 2)
  #     print(i, ids)

  from nbox.nbxlib.serving import serve_operator
  serve_operator(genimage, port = 9000)
