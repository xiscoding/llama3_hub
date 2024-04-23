# webgui: usage

## Parameters
3 tabs with options Generation, Chat, Instruction Template

### Generation:

### Chat: 

1. Character 

2. User

3. Chat history
    - Save chat history (.json)
    - upload chat histroy (.json)
4. Upload character

### Instruction Template (llama 3):
Each model has a specific format for instruction and chat templates
- Use the template below if you lose it
- Default template for llama 3 is not restored when new template is used.
```
{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = '<|begin_of_text|>' + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>
' }}{% endif %}
```

### Chat template (llama 3): 

```
{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        {%- if message['content'] -%}
            {{- message['content'] + '\n\n' -}}
        {%- endif -%}
        {%- if user_bio -%}
            {{- user_bio + '\n\n' -}}
        {%- endif -%}
    {%- else -%}
        {%- if message['role'] == 'user' -%}
            {{- name1 + ': ' + message['content'] + '\n'-}}
        {%- else -%}
            {{- name2 + ': ' + message['content'] + '\n' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
```

### Custom system message: 
from docs: 
`Custom system message: A message that defines the personality of the chatbot, replacing its default "System message" string. Example: "You are a duck."`

















## Prompting 
Negative Prompting: https://stable-diffusion-art.com/how-negative-prompt-work/