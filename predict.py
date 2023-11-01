import os
import time
import torch
import shutil
import socket
from PIL import Image
import urllib.request
from cog import BasePredictor, Input, Path
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from src.helpers import create_outpainting_image_and_mask
from src.weights import WeightsDownloadCache

VAE = ""
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_MODEL = ""

class Predictor(BasePredictor):

    def setup(self):
        start = time.time()
        print("Started setup")
        self.hf_token = "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"
        self.vae = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_type = torch.float32 if torch.is_floating_point(torch.tensor(32)) else torch.float16 
        self.variant = "fp32" if torch.is_floating_point(torch.tensor(32)) else "fp16"
        self.base_model_dir = ""
        self.refiner_model_dir = ""
        self.vae_model_dir = ""
        self.weights_cache = WeightsDownloadCache()
        print("setup took: ", time.time() - start)

    def isValid(url):
        try:
            socket.create_connection((url, 80))
            return True
        except socket.gaierror:
            return False
        except OSError:
            return False

    def load_image(self, path):
        # check if it's a remote url 
        if self.isValid(path):
            urllib.request.urlretrieve(path, "/tmp/image.png") 
        else:
            shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    

    #@torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Write the prompt here",
            default=None
        ),
        image_url: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            default=30
        ),
        seed: int = Input(
            description="Seed number",
            default=None
        )
    ) -> Path:
        
        if prompt is None:
            raise Exception("You should write a prompt")
        
        if image_url is None:
            raise Exception("You should provide an image")
       
        # Seed generator 
        if seed is None: 
            seed = torch.manual_seed(int.from_bytes(os.urandom(2), "big"))
        
        print("Starting... ")
        # Download the weights 
        self.base_model_dir = self.weights_cache.ensure(BASE_MODEL, { "token" : self.hf_token, "torch_type" : self.torch_type, "variant" : self.variant  })
        # self.refiner_model_dir = self.weights_cache.ensure(REFINER_MODEL)
        # self.vae_model_dir = self.weights_cache.ensure(VAE)

        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(self.base_model_dir, torch_dtype=self.torch_type, variant=self.variant)
        pipe.enable_model_cpu_offload()

        init_image = self.load_image(image_url)
        init_image = init_image.resize((512, 512))
        
        conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, 0.5)

        output = pipe(
            prompt,
            image=conditioning_image,
            mask_image=outpaint_mask,
            height=1024,
            width=1024,
            generator=seed,
            num_inference_steps=num_inference_steps
        ).images[0]

        return Path( output )
    
