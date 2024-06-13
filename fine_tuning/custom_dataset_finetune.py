from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

from datasets import load_dataset, Dataset
import pandas as pd

def formatting_prompts_func(examples):
    roles = examples["role"]
    contents = examples["content_chatML"]
    texts = [f"{role}: {content}" for role, content in zip(roles, contents)]
    return {"text": texts}

def create_hf_dataset_from_csv(csv_file_path):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Apply the formatting function
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return formatted_dataset

csv_file_path = "selected_chats.csv"  # The path to the CSV file created by export_selected_chats_to_csv
dataset = create_hf_dataset_from_csv(csv_file_path)

from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    output_dir="outputs"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args
)

trainer.train()
model.save_pretrained("path/to/save/model")
tokenizer.save_pretrained("path/to/save/model")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "path/to/save/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

input_text = "Your input prompt here"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=400)
predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(predicted_text)
