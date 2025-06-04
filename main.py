from huggingface_hub import login

from transformers import pipeline
import torch


ACCESS_TOKEN = "HUGGINGFACE_ACCESS_TOKEN_COMES_HERE"
login(ACCESS_TOKEN)

# https://huggingface.co/google/gemma-3-1b-it
MODEL = "google/gemma-3-1b-it"

pipe = pipeline(
    "text-generation",
    model=MODEL,
    torch_dtype=torch.bfloat16,
)

messages = [
    [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "소프트웨어 마에스트로에 관련된 시를 써줘"},
            ],
        },
    ],
]

output = pipe(messages, max_new_tokens=50)

"""Expected output:
[
    [
        {
            'generated_text': [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]
                }, 
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': '소프트웨어 마에스트로에 관련된 시를 써줘'}]
                },
                {
                    'role': 'assistant',
                    'content': '## 소프트웨어 마에스트로\n\n어둠 속, 코드의 춤\n새로운 세계를 창조하는 예술가\n소프트웨어 마에스트로, 그 이름은\n세상의 지식으로 빚어낸 섬세'
                }
            ]
        }
    ]
]

"""

print(output[0][0]["generated_text"][2])
