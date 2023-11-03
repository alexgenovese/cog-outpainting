import os, time, torch, shutil
from tqdm import tqdm
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL, DiffusionPipeline
from diffusers.utils import load_image
from src.helpers import create_outpainting_image_and_mask
from src.download_weights import cache_refiner, cache_base_model, cache_vae

VAE_CACHE = '/src/weights-cache/vae'
BASE_MODEL_CACHE = '/src/weights-cache/base_model'
REFINER_MODEL_CACHE = '/src/weights-cache/refiner_model'
BASE_MODEL = "alexgenovese/reica06"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0" # https://civitai.com/models/160350/not-real-realistic-xl
VAE = "madebyollin/sdxl-vae-fp16-fix"

temp_local_image = "/tmp/image.png"

class Predictor(BasePredictor):

    def get_torch_type(self):
        if torch.backends.mps.is_available():
            print("Torch MPS")
            return torch.float16
        
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16

            if torch.is_floating_point(torch.tensor(32)):
                return torch.float32
            
            if torch.is_floating_point(torch.tensor(16)):
                return torch.float16

        return torch.float16
    
    def get_device_type(self):
        if torch.backends.mps.is_available():
            print("MPS Device Type")
            return "mps"
        
        if torch.cuda.is_available():
            print("CUDA Device type")
            return "cuda"
        
        print("CPU Device type")
        return "cpu"
    
    def setup(self):
        start = time.time()
        print("Setup started...")
        with tqdm(total=100, desc="Setup") as pbar:
            self.hf_token = "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"
            self.device = self.get_device_type()
            self.torch_type = self.get_torch_type()
            self.variant = "fp32" if torch.is_floating_point(torch.tensor(32)) else "fp16"
            self.cache_base_model = BASE_MODEL_CACHE
            self.cache_refiner_model = REFINER_MODEL_CACHE
            self.cache_vae_model = VAE_CACHE
            self.in_base_model = None
            self.in_ref_model = None
            self.in_vae_model = None

            # check if cached models
            if not os.path.exists(self.cache_vae_model): 
                self.vae = cache_vae()
            pbar.update(20)

            if not os.path.exists(self.cache_base_model):
                cache_base_model()
            pbar.update(20)

            if not os.path.exists(self.cache_refiner_model):
                cache_refiner()
            pbar.update(20)

            # Instanciate references
            self.in_vae_model = AutoencoderKL.from_pretrained(VAE, torch_dtype=self.get_torch_type(), cache_dir=self.cache_vae_model )
            pbar.update(10)

            self.in_base_model = DiffusionPipeline.from_pretrained(
                BASE_MODEL,
                cache_dir=self.cache_base_model,
                torch_dtype=self.get_torch_type(),
                use_safetensors=False,
                vae=self.in_vae_model,
                add_watermark=False
            )
            pbar.update(10)

            self.in_ref_model = DiffusionPipeline.from_pretrained(
                REFINER_MODEL,
                cache_dir=self.cache_refiner_model,
                text_encoder_2=self.in_base_model.text_encoder_2,
                vae=self.in_vae_model,
                torch_dtype=self.get_torch_type(),
                use_safetensors=False,
                add_watermark=False
            )
            # refiner.watermark = NoWatermark() # remove base watermark
            pbar.update(10)

        print("setup took: ", time.time() - start)


    def load_image(self, path):
        shutil.copyfile(path, temp_local_image)
        return load_image(temp_local_image).convert("RGB")


    @torch.inference_mode()
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
        
        self.in_base_model = StableDiffusionXLInpaintPipeline.from_pretrained(self.in_base_model, cache_dir=self.cache_base_model, vae=self.in_vae_model, torch_dtype=self.get_torch_type(), variant=self.variant)
        self.in_base_model.enable_model_cpu_offload()

        init_image = self.load_image(image_url)
        init_image = init_image.resize((512, 512))
        
        conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, 0.5)

        output = self.in_base_model(
            prompt,
            image=conditioning_image,
            mask_image=outpaint_mask,
            height=1024,
            width=1024,
            generator=seed,
            num_inference_steps=num_inference_steps
        ).images[0]

        output.save(temp_local_image)

        return Path( temp_local_image )
    
