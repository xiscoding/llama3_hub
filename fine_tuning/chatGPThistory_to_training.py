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
    
def find_row(titile):
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


def save_df_to_file(df,  filename = "conversations_processed.jsonl", index = False):
    df.to_csv(filename, index= index)