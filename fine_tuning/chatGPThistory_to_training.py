#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:57:41 2024

@author: xdoestech
"""

import json
import sys
import random

with open('conversations.json') as file: 
    data = json.load(file)
    
import pandas as pd

df = pd.json_normalize(data)

def select_row_by_title(df, title_value):
    """Select and return row from DataFrame based on its 'title' value"""
    
    # Filter rows where 'title' matches the given value
    matching_rows = df[df['title'] == title_value]
    
    almost_matching_rows = df[title_value in df['title']]
    
    if not matching_rows.empty:
        return matching_rows.iloc[0], "matching"
    elif not almost_matching_rows.empty:
        return almost_matching_rows.iloc[0], "almost matching"
    else:
        return None
    
def find_row(title):
    title_to_find = title
    selected_row, match_type = select_row_by_title(df, title_to_find)
    
    if selected_row is not None: 
        print(f"found {match_type} row:")
        print(selected_row)
    else:
        print("No row found with title: ", title_to_find) 
                
def process_dataset(data):
    processed_data = []

    for entry in data:
        title = entry.get('title', "No Title")
        messages = []
        mapping = entry.get('mapping', {})

        for key, value in mapping.items():
            message_info = value.get('message')
            if message_info:
                role = message_info.get('author', {}).get('role')
                content = message_info.get('content', {}).get('parts', [])

                # Check for valid role and content, and ensure role is not 'system'
                if role and content and role != "system":
                    messages.append({"role": role, "content": content[0]})

        # Add entry if there are messages in the conversation
        if messages:
            processed_data.append({"title": title, "messages": messages})

    # Create and return the DataFrame directly
    return pd.DataFrame(processed_data)

def process_dataset_chatML(data):
    processed_data = []
    for entry in data:
        title = entry.get('title', "No Title")
        messages = []
        messages_chatML = ""  # Initialize the chatML string

        mapping = entry.get('mapping', {})
        for key, value in mapping.items():
            message_info = value.get('message')
            if message_info:
                role = message_info.get('author', {}).get('role')
                content = message_info.get('content', {}).get('parts', [])

                # Check for valid role, content, and role exclusion
                if role and content and role != "system":
                    messages.append({"role": role, "content": content[0]})

                    # Append message to the chatML string in the correct format
                    messages_chatML += f"<|im_start|>{role}\n{content[0]}<|im_end|>\n"

        # Add the 'assistant' prompt if needed in the chatML string
        if entry.get('add_generation_prompt', False):
            messages_chatML += "<|im_start|>assistant\n"

        # Add entry if there are messages in the conversation
        if messages:
            processed_data.append({
                "title": title,
                "messages": messages,  # Keep the original list for reference
                "messages_chatML": messages_chatML
            })

    return pd.DataFrame(processed_data)

def create_dataset_prompt_response(data):
    chat_history = []
    for entry in data:
        title = entry.get('title', "No Title")
        messages = []
        mapping = entry.get('mapping', {})
        for key, value in mapping.items():
            message_info = value.get('message')
            if message_info:
                role = message_info.get('author', {}).get('role')
                content = message_info.get('content', {}).get('parts', [])

                # Check for valid role, content, and role exclusion
                if role and content and role != "system":
                    # Handle content if it is a list of dicts
                    content_text = content[0].get('text', '') if isinstance(content[0], dict) else content[0]
                    
                    # Escape special characters
                    content_text = content_text.replace('\n', '\\n').replace('"', '\\"')
                    
                    # Format the message
                    messages_chatML = f"<|im_start|>{role}\n{content[0]}<|im_end|>\n"
                    new_record = {"chat_title": title, "role": role, "content": content_text, "content_chatML": messages_chatML}
                    messages.append(new_record)

        chat_history.append({"chat_title": title, "messages": messages})
    return pd.DataFrame(chat_history)

def export_selected_chats_to_csv(df, chat_titles, output_csv_path):
    # Filter the DataFrame to only include rows with the specified chat titles
    filtered_df = df[df['chat_title'].isin(chat_titles)]

    # Initialize a list to hold the rows for the new CSV
    csv_rows = []

    # Loop through the filtered DataFrame and extract the role and content_chatML pairs
    for index, row in filtered_df.iterrows():
        chat_title = row['chat_title']
        messages = row['messages']
        for message in messages:
            csv_rows.append({
                "chat_title": chat_title,
                "role": message['role'],
                "content_chatML": message['content_chatML']
            })

    # Create a new DataFrame from the extracted rows
    csv_df = pd.DataFrame(csv_rows)

    # Save the DataFrame to a CSV file
    csv_df.to_csv(output_csv_path, index=False)

def save_df_to_file(df,  filename = "conversations_chatML_June13_1", index = False, type="parquet"):
    if type ==  "parquet":
        df.to_parquet(f"{filename}.{type}", index=index)
    if type == "csv":
        df.to_csv(f"{filename}.{type}", index= index)
    if type == "json":
        df.to_json(f"{filename}.{type}", index= index)

if __name__ == '__main__':
    with open('conversations.json') as file: 
        data = json.load(file)
    df = create_dataset_prompt_response(data)
    print(df.head())
    print
    save_df_to_file(df)