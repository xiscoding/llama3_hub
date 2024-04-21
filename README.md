# llama3_hub
#LLAMA 3 Release 
### inital reaction (for historical purposes)
- Reddit inital opinion: https://www.reddit.com/r/LocalLLaMA/comments/1c79ci4/- meta_llama_3_is_out_first_impressions/
- Meta official page: https://ai.meta.com/blog/meta-llama-3/ 
- Linkedin Release: https://www.linkedin.com/pulse/meta-unveils-llama-3-cutting-edge-open-source-language-model-iqnjc/

## Weights (Hugging Face):
#### meta-llama official weights: 
- my google drive: https://drive.google.com/drive/folders/1aRaFzuu5NQyJiSEJcv_I5ZQCw_pMWra8
#### LoneStriker 8 bit weights: [LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2](https://huggingface.co/LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2/discussions/1) 
- [8b](https://huggingface.co/LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2) model
- [70b](https://huggingface.co/LoneStriker/Meta-Llama-3-70B-Instruct-2.25bpw-h6-exl2) model
- SYMBOLIC LINKING APPEARS BROKEN. Download the LoneStriker output.safetensors, tokenizer.json directly. 
#### NousResearch (GGUF): 
- [8b](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF)
- [70](https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF)
#### NousResearch (there are bunch)
- [link to page](https://huggingface.co/NousResearch)
#### Astronomer-io (8bit):
- [8b](https://huggingface.co/astronomer-io/Llama-3-8B-Instruct-GPTQ-8-Bit/tree/main)

## non-GUI interfaces
### Meta Llama3 run resources (torch)
- The official method of running llama (with unofficial weights)
- github: https://github.com/meta-llama/llama3
- [RECIPES](https://github.com/meta-llama/llama-recipes)
- - `pip install llama-recipes[tests, vllm]`
- SAFETY CONCERNS DOCUMENTATION:
- [Meta Llama Guard](https://github.com/meta-llama/llama-recipes/tree/main/recipes/responsible_ai)
- [meta trust and safety tools](https://llama.meta.com/trust-and-safety/)

### Hugging Face script (huggingface-cli)
- Python file inference
- - imports: [transformers](https://pypi.org/project/transformers/), [torch](https://pypi.org/project/torch/)
- **login to huggingface**
- - set up huggingface account
- - get token from [huggingface settings page](https://huggingface.co/settings/tokens)
- - run `huggingface-cli login` in terminal
- - Optional: set up [git-credential](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)

#### ISSUES
##### **ImportError: libcufft.so.10: cannot open shared object file: No such file or directory**
torch version is incompatable with cuda version
- `pip uninstall torch torchaudio torchvision`
- cuda version 12.1 (install latest version): `pip3 install torch torchvision torchaudio`
- - cuda version 11.8 (install older version): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
##### **while loading with LlamaForCausalLM, an error is thrown:**
The path to your model is most likely incorrect. 
- Ensure the model_id variable in your code matches the exact directory name where the downloaded model files are located.
- model path: LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2
##### **ValueError: Couldn't instantiate the backend tokenizer from one of:**
-make sure the tokenizer includes the model name not path
##### **bash: /home/xdoestech/.local/bin/huggingface-cli: No such file or directory**
- export PATH="/home/xdoestech/.local/bin:$PATH"
- NOT PERMANENT SOLUTION: may cause further errors if you have multiple environments with different versions of huggingface-cli

## GUI interfaces
### oobabooga
all models (probably, so far, your weird esoteric model type won't work)<br>
github: [link](https://github.com/oobabooga/text-generation-webui)
##### Installation: 
1. clone the repo into desired directory
2. open the start_{YOUR OPERATING SYSTEM}.sh script
    1. Check the config section and ensure paths are appropriate
    2. check the '# create the installer env' section and modify the "$INSTALL_ENV_DIR" variable if necessary
       - this will be the name of the conda environment that holds all dependencies for text generation web ui
    3.  Run ./start_{YOUR OPERATING SYSTEM}.sh
3. open the url that shows up in the terminal (Running on local URL:  http://127.0.0.1:7860)
##### Using Models: 
1. put model files into single directory in "text-generation-webui/models/"
2. Set up model in models tab in webgui 
    1. if needed fix symbolic links: https://www.freecodecamp.org/news/linux-ln-how-to-create-a-symbolic-link-in-linux-example-bash-command/
    2. Or just download directly from source
3. Refresh models next to load button, select model
4. Use correct model loader for model type (usually model type is in the name)
5. RUNNING astronmer-io GPTQ models IN OOBABOOGA (probably for all GPTQ models)
    1. Load the model via AutoGPTQ, with no_inject_fused_attention enabled. This is a bug with AutoGPTQ library.
    2. Change Parameters: Under Parameters -> Generation -> Skip special tokens: turn this off (deselect)
    3. Under Parameters -> Generation -> Custom stopping strings: add "<|end_of_text|>","<|eot_id|>" to the field

### lm studio
gguf models<br>
github: [link](https://lmstudio.ai/)
#### Installation: 
- just run the appImage from the website
### koboldcpp
gguf models<br>ggml models<br>
KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models.<br>
github: [link](https://github.com/LostRuins/koboldcpp)