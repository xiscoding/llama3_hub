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

## chatGPT history data format
``` json
sample_data = [
    {
        "title": "Kivy Workout App Transition",
        "create_time": 1717275611.245864,
        "update_time": 1717343856.547421,
        "mapping": {
            "b2606240-aab9-48fc-9810-48e63ff1a7f4": {
                "id": "b2606240-aab9-48fc-9810-48e63ff1a7f4",
                "message": {
                    "id": "b2606240-aab9-48fc-9810-48e63ff1a7f4",
                    "author": {"role": "system", "name": None, "metadata": {}},
                    "create_time": None,
                    "update_time": None,
                    "content": {"content_type": "text", "parts": [""]},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 0.0,
                    "metadata": {"is_visually_hidden_from_conversation": True},
                    "recipient": "all"
                },
                "parent": "aaa1c7a0-8abc-420c-89d0-769df8dca8a8",
                "children": ["aaa25b5c-4c8a-476a-9bbf-a00bef55673a"]
            },
            "aaa1c7a0-8abc-420c-89d0-769df8dca8a8": {
                "id": "aaa1c7a0-8abc-420c-89d0-769df8dca8a8",
                "message": None,
                "parent": None,
                "children": ["b2606240-aab9-48fc-9810-48e63ff1a7f4"]
            },
            "aaa25b5c-4c8a-476a-9bbf-a00bef55673a": {
                "id": "aaa25b5c-4c8a-476a-9bbf-a00bef55673a",
                "message": {
                    "id": "aaa25b5c-4c8a-476a-9bbf-a00bef55673a",
                    "author": {"role": "user", "name": None, "metadata": {}},
                    "create_time": 1717275611.281665,
                    "update_time": None,
                    "content": {"content_type": "text", "parts": ["You are going to create a workout app for android. You will use kivy, buildozer, and python. I will provide 2 example files for you to use. One file is the main logic for the workout app, the other file is an example version of the app using tkinter. You will use both files to create a kivy version. "]},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 0.0,
                    "metadata": {},
                    "recipient": "all"
                },
                "parent": "aaa1c7a0-8abc-420c-89d0-769df8dca8a8",
                "children": []
            }
        }
    }
]
```
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
