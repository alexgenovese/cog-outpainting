import os, time, torch, shutil
from tqdm import tqdm
from PIL import Image
from compel import Compel
from huggingface_hub import login
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL, DiffusionPipeline
from diffusers.utils import load_image
from src.helpers import create_outpainting_image_and_mask
from src.download_weights import cache_refiner, cache_base_model, cache_vae
from src.prompt import PromptManager

VAE_CACHE = './weights-cache/vae'
BASE_MODEL_CACHE = './weights-cache/base_model'
REFINER_MODEL_CACHE = './weights-cache/refiner_model'
BASE_MODEL = "alexgenovese/reica06"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0" # https://civitai.com/models/160350/not-real-realistic-xl
VAE = "madebyollin/sdxl-vae-fp16-fix"

temp_local_image = "./image.png"

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
            self.prompt_manager = PromptManager(Compel)

            print(f"Settings {self.variant} {self.torch_type} {self.device}")

            # Login in HF
            login( token = self.hf_token )

            # check if cached models
            if not os.path.exists(self.cache_vae_model): 
                self.vae = cache_vae()
            pbar.update(20)

            if not os.path.exists(self.cache_base_model):
                cache_base_model()
            pbar.update(20)

            # if not os.path.exists(self.cache_refiner_model):
                # cache_refiner()
            pbar.update(20)

            # Instanciate references
            print(f"Setup VAE {self.cache_vae_model} {VAE}")
            self.in_vae_model = AutoencoderKL.from_pretrained(VAE, torch_dtype=self.torch_type, cache_dir=self.cache_vae_model )
            pbar.update(10)

            print(f"Setup BASE {self.cache_base_model} {BASE_MODEL}")
            self.in_base_model = StableDiffusionXLInpaintPipeline.from_pretrained(
                BASE_MODEL,
                cache_dir=self.cache_base_model,
                torch_dtype=self.torch_type,
                vae=self.in_vae_model,
                add_watermark=False
            )
            self.in_base_model.to(self.device)
            pbar.update(10)
            
            """
            print(f"Setup REFINER {self.cache_refiner_model} {REFINER_MODEL}")
            self.in_ref_model = DiffusionPipeline.from_pretrained(
                REFINER_MODEL,
                cache_dir=self.cache_refiner_model,
                text_encoder_2=self.in_base_model.text_encoder_2,
                vae=self.in_vae_model,
                torch_dtype=self.torch_type,
                add_watermark=False
            )
            # refiner.watermark = NoWatermark() # remove base watermark
            """

            pbar.update(10)

        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, temp_local_image)
        return load_image(temp_local_image).convert("RGB")

    def predict_mps(self,
                    prompt: str = "",
                    negative_prompt: str = "(((golden ratio))), bw, (tan skin:1.3),(worst quality:2), (low quality:2), low-res, (nose2), (((chromatic aberration))), ((blur censor)), ((blurry)), (blurry background), (blurry foreground), bokeh, (chromatic aberration), cosplay photo, eyelashes, motion blur, nose, overexposed, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, 4 fingers, 3 fingers, too many fingers, long neck, open mouth, closed mouth",
                    num_inference_steps: int = 30,
                    image_url: str = "00041-3204191605.png",
                    seed: str = 3204191605
        ):

        print(f"Starting {self.variant} {self.torch_type} {self.device}")

        if seed is None: 
            seed = torch.manual_seed(int.from_bytes(os.urandom(2), "big"))
        else:
            seed = torch.manual_seed(seed)

        # self.in_base_model.enable_model_cpu_offload()

        prompt = self.prompt_manager.rewrite_prompt_for_compel(prompt)
        negative_prompt = self.prompt_manager.rewrite_prompt_for_compel(negative_prompt)

        # Define Compel for Base model 
        conditioning, pooled, neg_conditioning, neg_pooled = self.prompt_manager.createCompel(self.in_base_model, prompt, negative_prompt)
        # Define compel for Refiner model 
        # ref_conditioning, ref_pooled, ref_neg_conditioning, ref_neg_pooled = self.prompt_manager.createCompel(self.in_ref_model, prompt, negative_prompt, [True])


        k = 1
        for i in [0.9, 0.7, 0.5]: 
            
            init_image = self.load_image(image_url)
            init_image = init_image.resize((512, 512))    
            conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, i)

            output = self.in_base_model(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                image=conditioning_image,
                mask_image=outpaint_mask,
                height=1024,
                width=1024,
                generator=seed,
                num_inference_steps=num_inference_steps
            ).images[0]

            output_image_link = f"./zoom_out_{k}.png"
            output.save(f"{output_image_link}")
            k = k + 1
            image_url = output_image_link
            

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Write the prompt here",
            default="(Realistic Photo:2) of (Ultra detailed:1.8) glamour portrait shot (from above:0.5) of rich sophisticated old european in fancy clothes, ((overwhelming fatigue)), wrinkles of age, photorealistic, moody colors, gritty, masterpiece, best quality, (intricate details), (****), eldritch, glow, glowing eyes, (volumetric lighting), unique pose, dynamic pose, dutch angle, 35mm, anamorphic, lightroom, cinematography, film grain, HDR10, 8k hdr, Steve McCurry, ((cinematic)), RAW, color graded portra 400 film, remarkable color, raytracing, subsurface scattering, hyperrealistic, extreme skin details, skin pores, deep shadows, contrast, dark theme"
        ),
        negative_prompt: str = Input(
            description="Write the prompt here",
            default="(((golden ratio))), bw, (tan skin:1.3),(worst quality:2), (low quality:2), low-res, (nose2), (((chromatic aberration))), ((blur censor)), ((blurry)), (blurry background), (blurry foreground), bokeh, (chromatic aberration), cosplay photo, eyelashes, motion blur, nose, overexposed, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, 4 fingers, 3 fingers, too many fingers, long neck, open mouth, closed mouth"
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
        
        print(f"Starting {self.variant} {self.torch_type} {self.device}")
        self.in_base_model.enable_model_cpu_offload()

        prompt = self.prompt_manager.rewrite_prompt_for_compel(prompt)
        negative_prompt = self.prompt_manager.rewrite_prompt_for_compel(negative_prompt)

        # Define Compel for Base model 
        conditioning, pooled, neg_conditioning, neg_pooled = self.prompt_manager.createCompel(self.in_base_model, prompt, negative_prompt)
        # Define compel for Refiner model 
        # ref_conditioning, ref_pooled, ref_neg_conditioning, ref_neg_pooled = self.prompt_manager.createCompel(self.in_ref_model, prompt, negative_prompt, [True])


        init_image = self.load_image(image_url)
        init_image = init_image.resize((768, 768))
        
        conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, 0.5)

        output = self.in_base_model(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_conditioning,
            negative_pooled_prompt_embeds=neg_pooled,
            image=conditioning_image,
            mask_image=outpaint_mask,
            height=1024,
            width=1024,
            generator=seed,
            num_inference_steps=num_inference_steps
        ).images[0]

        output.save(temp_local_image)

        return Path( temp_local_image )
    

def main():
    pred = Predictor()
    pred.setup()
    pred.predict_mps()


if __name__ == "__main__":
    main()