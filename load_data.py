# %% [markdown]
# # Load GPT and GPT 2

# %%
from transformers import pipeline, set_seed 
generation_gpt = pipeline("text-generation", model="openai-gpt")
generation_gpt2 = pipeline("text-generation", model="gpt2")

# %%
def model_size(model):
    return sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size(generation_gpt.model)/1000**2:.1f}M parameters")
print(f"GPT2 size: {model_size(generation_gpt2.model)/1000**2:.1f}M parameters")


# %%
def enum_pipeline_output(pipe, prompt, num_return_sequences): 
    out = pipe(prompt, num_return_sequences=num_return_sequences,
               clean_up_tokenization_spaces=True)
    return "\n".join(f"{i+1}." + x["generated_text"] for i, x in enumerate(out))
prompt = "\nwhen they came back"
print("GPT:\n" + enum_pipeline_output(generation_gpt, prompt, 3))
print()
print("GPT2:\n" + enum_pipeline_output(generation_gpt2, prompt, 3))

# %% [markdown]
# # Load data

# %%
import os

# %%
# # add all file in /mnt/data3/nghiaph/pull-github/pull-github-bucket-2109-ver3 to .json file 
# folder_path = "/mnt/data3/nghiaph/pull-github/pull-github-bucket-2109-ver3/" 
# for file in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file)
#     new_file_path = os.path.join(folder_path, file + '.json')
#     os.rename(file_path, new_file_path)

# print("All files have been renamed with the .json extension.")


# %%
os.environ['TRANSFORMERS_CACHE'] = '/mnt/data3/nghiaph/huggingface/transformers'
print(os.getenv('TRANSFORMERS_CACHE'))


# %%


# %%
from datasets import load_dataset, DownloadConfig
download_config = DownloadConfig(delete_extracted=True,
                                 cache_dir='/mnt/data3/nghiaph/huggingface/transformers/datasets') 
download_config.cache_dir

# %%
dataset = load_dataset("/mnt/data3/nghiaph/pull-github/pull-github-bucket-2109-ver3/", 
                       split="train", download_config=download_config)

# %%
import psutil
import os
print(f"Number of python files code in dataset : {len(dataset)}")
ds_size = sum(os.stat(f["filename"]).st_size for f in dataset.cache_files)
# os.stat.st_size is expressed in bytes, so we convert to GB
print(f"Dataset size (cache file) : {ds_size / 2**30:.2f} GB")
# Process.memory_info is expressed in bytes, so we convert to MB
print(f"RAM used: {psutil.Process(os.getpid()).memory_info().rss >> 20} MB")

# %%
from huggingface_hub import scan_cache_dir
delete_strategy = scan_cache_dir().delete_revisions()
print("Will free " + delete_strategy.expected_freed_size_str)

# %%



