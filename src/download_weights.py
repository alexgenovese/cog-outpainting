import os, subprocess, time, shutil
from tqdm import tqdm
from huggingface_hub import login
from diffusers import AutoencoderKL,  StableDiffusionXLPipeline, DiffusionPipeline

VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
VAE_CACHE = '/content/weights-cache/vae'
BASE_MODEL = "alexgenovese/reica06"
BASE_MODEL_CACHE = '/content/weights-cache/base_model'
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0" # https://civitai.com/models/160350/not-real-realistic-xl
REFINER_MODEL_CACHE = '/content/weights-cache/refiner_model'
hf_token = "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"

pipe = None
vae = None
logged_in = False 

def login_hf():
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
            pipe = StableDiffusionXLPipeline.from_pretrained( BASE_MODEL )
            pipe.save_pretrained(BASE_MODEL_CACHE, safe_serialization=True)
            print("Downloading took: ", time.time() - start)
        except Exception as error:
            print("BASE_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(BASE_MODEL_CACHE)
            print("BASE_MODEL - Removed empty cache directory")


def cache_refiner(pipe):
    if not os.path.exists(REFINER_MODEL_CACHE):
        try: 
            os.makedirs(REFINER_MODEL_CACHE)

            if pipe is None:
                try: 
                  pipe = StableDiffusionXLPipeline.from_pretrained( BASE_MODEL, cache_dir=BASE_MODEL_CACHE )
                except Exception as error:
                  raise Exception("No base model found")

            refiner = DiffusionPipeline.from_pretrained(REFINER_MODEL)
            refiner.save_pretrained(REFINER_MODEL_CACHE, safe_serialization=True)
        except Exception as error:
            print("REFINER_MODEL - Something went wrong while downloading")
            print(f"{error}")
            shutil.rmtree(REFINER_MODEL_CACHE)
            print("REFINER_MODEL - Removed empty cache directory")



if __name__ == "__main__":
    print("-----> Start caching models...")
    with tqdm(total=100, desc="Creating cache") as pbar:
        login_hf()
        pbar.update(25)

        cache_vae(vae)
        pbar.update(25)

        cache_base_model(pipe)
        pbar.update(25)

        cache_refiner(pipe)
        pbar.update(25)

    print("-----> Caching completed!")