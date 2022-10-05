# this file contains all the code for running the stable diffusion model and all of it's APIs

import os
import PIL
import json
import numpy as np
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import PNDMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


def threaded_map(fn, inputs, wait: bool = True, max_threads = 20, _name: str = "threaded_map"):
  """convinience function that runs fn parallely with inputs and returns in asked order"""
  results = [None for _ in range(len(inputs))]
  max_threads = min(len(inputs), max_threads)
  with ThreadPoolExecutor(max_workers = max_threads, thread_name_prefix = _name) as exe:
    _fn = lambda i, x: [i, fn(*x)]
    futures = {exe.submit(_fn, i, x): i for i, x in enumerate(inputs)}
    if not wait:
      return futures
    for future in as_completed(futures):
      try:
        i, res = future.result()
        results[i] = res
      except Exception as e:
        raise e
  return results


def load_sub_models(
  pretrained_model_name_or_path,
  cache_dir: str = None,
  resume_download: bool = False,
  proxies: dict = None,
  local_files_only: bool = False,
  use_auth_token: str = None,
  revision: str = None,
  torch_dtype: str = None,
  enable_safety_module: bool = False
):
  """Function that replaces the from_pretrained in pipeline, responsible for downloading and loading of the sub models."""

  # 1. Download the checkpoints and configs
  # use snapshot download here to get it working from from_pretrained
  if not os.path.isdir(pretrained_model_name_or_path):
    cached_folder = snapshot_download(
      pretrained_model_name_or_path,
      cache_dir=cache_dir,
      resume_download=resume_download,
      proxies=proxies,
      local_files_only=local_files_only,
      use_auth_token=use_auth_token,
      revision=revision,
    )
  else:
    cached_folder = pretrained_model_name_or_path

  # 2. load the models one by one
  init_dict = {
    "vae": AutoencoderKL,
    "text_encoder": CLIPTextModel,
    "tokenizer": CLIPTokenizer,
    "unet": UNet2DConditionModel,
  }
  if enable_safety_module:
    init_dict.update({
      "safety_checker": StableDiffusionSafetyChecker,
      "feature_extractor": CLIPFeatureExtractor,
    })

  # 3. Load each module in the pipeline
  modules = threaded_map(
    lambda name, class_obj: (
      name,
      class_obj.from_pretrained(os.path.join(cached_folder, name))
    ),
    init_dict.items()
  )
  init = {k:v for k,v in modules}
  with open(f"{cached_folder}/scheduler/scheduler_config.json", "r") as f:
    init["scheduler"] = PNDMScheduler(**json.load(f))
  return init


class StableDiff:
  def __init__(
    self,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    scheduler: PNDMScheduler,
    safety_checker: StableDiffusionSafetyChecker = None,
    feature_extractor: CLIPFeatureExtractor = None,
  ):
    scheduler = scheduler.set_format("pt")
    self.vae = vae
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer
    self.unet = unet
    self.scheduler = scheduler
    self.safety_checker = safety_checker
    self.feature_extractor = feature_extractor
    self.set_seed()

  def to(self, device):
    args = [(self.vae,), (self.text_encoder,), (self.unet,)]
    if self.safety_checker is not None:
      args += [(self.safety_checker,), (self.feature_extractor,)]
    threaded_map(lambda mod: mod.to(device), args)
    self.device = device

  @staticmethod
  def numpy_to_pil(images):
    if images.ndim == 3:
      images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [PIL.Image.fromarray(image) for image in images]
    return pil_images

  @staticmethod
  def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

  @staticmethod
  def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask

  def set_seed(self, seed: int = 4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    self._gen = torch.Generator(seed)

  # the methods that generate the images

  @torch.no_grad()
  def img2img(
    self,
    prompt,
    init_image,
    strength = 0.8,
    num_inference_steps = 50,
    guidance_scale  = 7.5,
    seed: int = None,
  ):
    if isinstance(prompt, str):
      batch_size = 1
    elif isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    if strength < 0 or strength > 1:
      raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
    if num_inference_steps < 0:
      raise ValueError(f"The value of num_inference_steps should be positive but is {num_inference_steps}")
    if seed is not None:
      self.set_seed(seed)

    offset = 1
    self.scheduler.set_timesteps(num_inference_steps, offset = offset)

    # encode the init image into latents and scale the latents
    init_image = self.preprocess_image(init_image).to(self.device)
    init_latent_dist = self.vae.encode(init_image.to(self.device)).latent_dist
    init_latents = init_latent_dist.sample(generator = self._gen)
    init_latents = 0.18215 * init_latents

    # expand init_latents for batch_size
    init_latents = torch.cat([init_latents] * batch_size)

    # get the original timestep using init_timestep
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    timesteps = self.scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)    

    # add noise to latents using the timesteps
    noise = torch.randn(init_latents.shape, device=self.device, generator = self._gen)
    init_latents = self.scheduler.add_noise(init_latents, noise, timesteps).to(self.device)

    return self.forward(
      prompt = prompt,
      latents = init_latents,
      timesteps = self.scheduler.timesteps[max(num_inference_steps - init_timestep + offset, 0):],
      guidance_scale = guidance_scale,
      noise = noise,
    )

  @torch.no_grad()
  def inpaint(
    self,
    prompt,
    init_image,
    mask_image,
    strength = 0.8,
    num_inference_steps = 50,
    guidance_scale  = 7.5,
    seed: int = None,
  ):
    if isinstance(prompt, str):
      batch_size = 1
    elif isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    if strength < 0 or strength > 1:
      raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
    if num_inference_steps < 0:
      raise ValueError(f"The value of num_inference_steps should be positive but is {num_inference_steps}")
    if seed is not None:
      self.set_seed(seed)

    offset = 1
    self.scheduler.set_timesteps(num_inference_steps, offset = offset)

    # encode the init image into latents and scale the latents
    init_image = self.preprocess_image(init_image).to(self.device)
    init_latent_dist = self.vae.encode(init_image.to(self.device)).latent_dist
    init_latents = init_latent_dist.sample(generator = self._gen)
    init_latents = 0.18215 * init_latents

    # Expand init_latents for batch_size
    init_latents = torch.cat([init_latents] * batch_size)
    init_latents_orig = init_latents

    # preprocess mask
    mask = self.preprocess_mask(mask_image).to(self.device)
    mask = torch.cat([mask] * batch_size)

    # check sizes
    if not mask.shape == init_latents.shape:
      raise ValueError("The mask and init_image should be the same size!")

    # get the original timestep using init_timestep
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    timesteps = self.scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

    # add noise to latents using the timesteps
    noise = torch.randn(init_latents.shape, device = self.device, generator = self._gen)
    init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

    return self.forward(
      prompt = prompt,
      latents = init_latents,
      timesteps = self.scheduler.timesteps[max(num_inference_steps - init_timestep + offset, 0):],
      guidance_scale = guidance_scale,

      # things specific to inpainting
      mask = mask,
      orig_latents = init_latents_orig,
      noise = noise,
    )


  @torch.no_grad()
  def __call__(
    self,
    prompt = "a monkey sitting in spaceship through time, greg rutwowski, 4k, dmt",
    height = 512,
    width = 512,
    guidance_scale = 7.5,
    offset = 1,
    num_inference_steps = 50,
    latents = None,
    seed: int = None,
  ):
    if height % 8 != 0 or width % 8 != 0:
      raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    self.scheduler.set_timesteps(num_inference_steps, offset = offset)
    if isinstance(prompt, str):
      batch_size = 1
    elif isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    if seed is not None:
      self.set_seed(seed)
    
    # Unlike in other pipelines, latents need to be generated in the target device
    # for 1-to-1 results reproducibility with the CompVis implementation.
    # However this currently doesn't work in `mps`.
    _unet_dev = self.unet.device
    latents_device = "cpu" if _unet_dev.type == "mps" else _unet_dev
    latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
    if latents is None:
      latents = torch.randn(latents_shape, device=latents_device, generator = self._gen)
    else:
      if latents.shape != latents_shape:
        raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

    return self.forward(
      prompt = prompt,
      latents = latents,
      timesteps = self.scheduler.timesteps,
      guidance_scale = guidance_scale,
    )

  @torch.no_grad()
  def forward(
    self,
    prompt: str,
    latents,
    timesteps,
    guidance_scale = 7.5,
    mask = None,
    orig_latents = None,
    noise = None,
  ):
    if isinstance(prompt, str):
      batch_size = 1
    elif isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if mask is not None:
      assert orig_latents is not None
      assert noise is not None

    # get prompt text embeddings
    # text_input = {input_ids: 2D, attention_mask: 2D}
    text_input = self.tokenizer(
      prompt,
      padding="max_length",
      max_length=self.tokenizer.model_max_length,
      truncation=True,
      return_tensors="pt",
    )
    text_embeddings = self.text_encoder(text_input.input_ids.to(self.text_encoder.device))[0]

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
      max_length = text_input.input_ids.shape[-1]
      uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
      uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.text_encoder.device))[0]

      # For classifier free guidance, we need to do two forward passes.
      # Here we concatenate the unconditional and text embeddings into a single batch
      # to avoid doing two forward passes
      text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = latents.to(self.device)
    _pbar = trange(len(timesteps))
    for i, t in zip(_pbar, timesteps):
      # expand the latents if we are doing classifier free guidance
      latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

      # predict the noise residual
      noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

      # perform guidance
      if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      # compute the previous noisy sample x_t -> x_t-1
      latents = self.scheduler.step(noise_pred, t, latents).prev_sample
      if mask is not None:
        # this is the case for inpainting
        init_latents_proper = self.scheduler.add_noise(orig_latents, noise, t)
        latents = (init_latents_proper * mask) + (latents * (1 - mask))

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return self.numpy_to_pil(image)


def get_model(device = "cuda:0"):
  print("loading sub models")
  cls_init = load_sub_models(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token = "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",
    cache_dir = "../cache",
  )

  print(f"moving to {device}")
  stdif = StableDiff(**cls_init)
  stdif.to(device)
  return stdif
