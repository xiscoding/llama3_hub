# llama3_hub
#LLAMA 3 Release 
### inital reaction (for historical purposes)
- Reddit inital opinion: https://www.reddit.com/r/LocalLLaMA/comments/1c79ci4/- meta_llama_3_is_out_first_impressions/
- Meta official page: https://ai.meta.com/blog/meta-llama-3/ 
- Linkedin Release: https://www.linkedin.com/pulse/meta-unveils-llama-3-cutting-edge-open-source-language-model-iqnjc/

## Weights (Hugging Face):
#### meta-llama official weights: 
#### 8 bit weights: [LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2](https://huggingface.co/LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2/discussions/1) 
- [8b](https://huggingface.co/LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2) model
- [70b](https://huggingface.co/LoneStriker/Meta-Llama-3-70B-Instruct-2.25bpw-h6-exl2) model

## Running Model locally
### Meta Llama3 model Page
- The official method of running llama (with unofficial weights)
- github: https://github.com/meta-llama/llama3
- [RECIPES](https://github.com/meta-llama/llama-recipes)
- - `pip install llama-recipes[tests, vllm]`
- SAFETY CONCERNS DOCUMENTATION:
- [Meta Llama Guard](https://github.com/meta-llama/llama-recipes/tree/main/recipes/responsible_ai)
- [meta trust and safety tools](https://llama.meta.com/trust-and-safety/)

### Hugging Face script
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

### oobabooga
all models (probably, so far, your weird esoteric model type won't work)<br>
github: [link](https://github.com/oobabooga/text-generation-webui)
### lm studio
gguf models<br>
github: [link](https://lmstudio.ai/)
### koboldcpp
gguf models<br>ggml models<br>
KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models.<br>
github: [link](https://github.com/LostRuins/koboldcpp)