from huggingface_hub import snapshot_download
from pathlib import Path
import os

#Your token has been saved to /home/fengjun/.cache/huggingface/token
#Login successful

download_to = "~/.cache/huggingface/hub/"
os.makedirs(download_to, exist_ok=True)
# mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
# mistral_models_path.mkdir(parents=True, exist_ok=True)

# allow_pattens = [
#     'config.json',
#     'generation_config.json',
#     'model-00001-of-00003.safetensors',
#     'model-00002-of-00003.safetensors',
#     'model-00003-of-00003.safetensors',
#     'params.json',
#     'model.safetensors.index.json',
#     'special_tokens_map.json',
#     'tokenizer.json',
#     'tokenizer_config.json',
#     'tokenizer.model'
# ]
snapshot_download(repo_id="google/t5-3b", local_dir=download_to)