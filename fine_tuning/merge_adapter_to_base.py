import json
from huggingface_hub import login

config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
login(token = HF_TOKEN)

NEW_MODEL_NAME = "llama-3-8b-Instruct-July8-1"
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
new_model = f"/home/xdoestech/llama3_hub/{NEW_MODEL_NAME}"

###################################################################################
# merge base model with adapter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format
# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "20GB", 1: "10GB", 2: "0GB"} # xdoestech: 0:3090, 1:3060, 2:960
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(base_model_reload, new_model)

model = model.merge_and_unload()

###################################################################################
# Confirm functionality with model inference
TEST_QUERY = "Hello doctor, I have bad acne. How do I get rid of it?"
messages = [{"role": "user", "content": TEST_QUERY}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])

###################################################################################
# save model, total size ~ 16gb
model.save_pretrained("llama-3-8b-chat-doctor")
tokenizer.save_pretrained(NEW_MODEL_NAME)

###################################################################################
# push model to hugging face hub
model.push_to_hub("llama-3-8b-chat-doctor", use_temp_dir=False)
tokenizer.push_to_hub(NEW_MODEL_NAME, use_temp_dir=False)