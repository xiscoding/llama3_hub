from datasets import load_dataset, Dataset
import pandas as pd

def formatting_prompts_func(examples):
    roles = examples["role"]
    contents = examples["content_chatML"]
    texts = [f"{role}: {content}" for role, content in zip(roles, contents)]
    return {"text": texts}

def create_dataset_chatGPT(csv_file_path):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Apply the formatting function
    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return formatted_dataset

def create_dataset_doctor(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.shuffle(seed=65).select(range(1000)) # Only use 1000 samples for quick demo

    def format_chat_template(row):
        row_json = [{"role": "user", "content": row["Patient"]},
                {"role": "assistant", "content": row["Doctor"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=4,
    )

    # split dataset into training and validation
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset

if __name__ == '__main__':
    csv_file_path = "selected_chats.csv"  # The path to the CSV file created by export_selected_chats_to_csv
    dataset = create_hf_dataset_from_csv(csv_file_path)