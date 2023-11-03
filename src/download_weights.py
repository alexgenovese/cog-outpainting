import os, subprocess, time, shutil
from tqdm import tqdm
from huggingface_hub import login
from diffusers import AutoencoderKL,  StableDiffusionXLPipeline, DiffusionPipeline

VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
VAE_CACHE = '/src/weights-cache/vae'
# BASE_MODEL = "https://reica.s3.eu-south-1.amazonaws.com/__INTERNAL__/models/reica_06.safetensors"
BASE_MODEL = "alexgenovese/reica06"
BASE_MODEL_CACHE = '/src/weights-cache/base_model'
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0" # https://civitai.com/models/160350/not-real-realistic-xl
REFINER_MODEL_CACHE = '/src/weights-cache/refiner_model'
hf_token = "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"

pipe = None
vae = None
logged_in = False 

def login():
    if logged_in is False:
        login( token = hf_token )
    
    return True


def cache_vae():
    if not os.path.exists(VAE_CACHE): 
        try:
            os.makedirs(VAE_CACHE)
            vae = AutoencoderKL.from_pretrained(VAE_MODEL)
            vae.save_pretrained(VAE_CACHE, safe_serialization=True)
        except Exception as error:
            print("VAE - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(VAE_CACHE)
            print("VAE - Removed empty cache directory")

def cache_base_model():
    if not os.path.exists(BASE_MODEL_CACHE):
        try:
            os.makedirs(BASE_MODEL_CACHE)
            start = time.time()
            print("Converting Reica custom model")
            pipe = StableDiffusionXLPipeline.from_pretrained(BASE_MODEL, vae = vae, use_safetensors=True)
            pipe.save_pretrained(BASE_MODEL_CACHE, safe_serialization=True)
            print("Downloading took: ", time.time() - start)
        except Exception as error:
            print("BASE_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(BASE_MODEL_CACHE)
            print("BASE_MODEL - Removed empty cache directory")


def cache_refiner():
    if not os.path.exists(REFINER_MODEL_CACHE):
        try: 
            os.makedirs(REFINER_MODEL_CACHE)

            if pipe is None:
                raise Exception("No base model found")

            refiner = DiffusionPipeline.from_pretrained(
                REFINER_MODEL,
                text_encoder_2=pipe.text_encoder_2,
                vae=vae,
                use_safetensors=True,
                variant="fp16"
            )
            refiner.save_pretrained(REFINER_MODEL_CACHE, safe_serialization=True)
        except Exception as error:
            print("REFINER_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(REFINER_MODEL_CACHE)
            print("REFINER_MODEL - Removed empty cache directory")


if __name__ == "__main__":
    print("-----> Start caching models...")
    with tqdm(total=100, desc="Creating cache") as pbar:
        login()
        pbar.update(25)

        cache_vae()
        pbar.update(25)

        cache_base_model()
        pbar.update(25)

        cache_refiner()
        pbar.update(25)

    print("-----> Caching completed!")