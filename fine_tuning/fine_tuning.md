# Fine Tune Llama3 

Overview: The process here is using huggingface hub. There should be little change if you decide to use a different model, dataset, etc. 

## Process overview: 
1. Create Dataset (NOT IMPLEMENTED YET, similar to other code in gpt_trainer repo)
2. train adapter
3. merge adapter
4. FIN!

BEST SOURCES: 
1. https://www.datacamp.com/tutorial/llama3-fine-tuning-locally
2. https://github.com/Zjh-819/LLMDataHub
3. https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html
4. https://huggingface.co/docs/trl/main/en/orpo_trainer#expected-dataset-format




# HuggingFace Datasets: 
* ORPO: 
    * https://huggingface.co/docs/trl/main/en/orpo_trainer#expected-dataset-format
    * Tutorial: https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi?usp=sharing

# Custom Datasets
* ChatML: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chat-markup-language
* Chat Templating HuggingFace : https://huggingface.co/docs/transformers/main/en/chat_templating
* tokenizer.apply_chat_template

# ISSUES:
* NotImplementedError: Cannot copy out of meta tensor; no data!
    * https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13087
    * Solution: Your GPUs are not compatible. Your GPU is out of memory
    * set max_memory={0: "20GB", 1: "10GB", 2: "0GB"} (WILL VARY DEPENDING ON DEVICE)
* OUT OF MEMORY
    * Solution: Get more vram, ensure your device is actually using the GPU
    * accelerate tutorial: https://www.youtube.com/watch?v=MWCSGj9jEAo
* ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM
    * Improper allocation of data to devices
    * set max_memory={0: "20GB", 1: "10GB", 2: "0GB"} (WILL VARY DEPENDING ON DEVICE)
* GENERAL ERROR WITH MEMORY
    * either your device is not using the device you want or improper mapping of data
    * https://github.com/huggingface/transformers/issues/24965
    * https://github.com/TimDettmers/bitsandbytes/issues/627
    * https://stackoverflow.com/questions/77713051/why-does-my-device-map-auto-in-transformers-pipline-uses-cpu-only-even-though
    * https://github.com/huggingface/transformers/issues/22595
