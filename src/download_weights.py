import os, subprocess, time, shutil
from tqdm import tqdm
from src.downloader import download
from diffusers import AutoencoderKL,  StableDiffusionXLPipeline, DiffusionPipeline

VAE_CACHE = '/src/weights-cache/vae'
BASE_MODEL_CACHE = '/src/weights-cache/base_model'
REFINER_MODEL_CACHE = '/src/weights-cache/refiner_model'
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
BASE_MODEL = "alexgenovese/reica06"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0" # https://civitai.com/models/160350/not-real-realistic-xl
hf_token = "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"

pipe = None
vae = None

def cache_vae():
    if not os.path.exists(VAE_CACHE): 
        try:
            os.makedirs(VAE_CACHE)
            vae = AutoencoderKL.from_pretrained(VAE_MODEL, use_safetensors=True)
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
            # download the file in the cache
            start = time.time()
            print("Downloading Reica custom model")
            file_cache = f"{BASE_MODEL_CACHE}/reica_06.safetensors"
            download(BASE_MODEL, file_cache)
            print("Downloading took: ", time.time() - start)
            
            # read local file
            start = time.time()
            print("Converting Reica custom model")
            pipe = StableDiffusionXLPipeline.from_single_file(file_cache, use_safetensors=True)
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
        cache_vae()
        pbar.update(33)

        cache_base_model()
        pbar.update(34)

        cache_refiner()
        pbar.update(33)

    print("-----> Caching completed!")