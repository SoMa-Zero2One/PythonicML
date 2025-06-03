from huggingface_hub import login

from transformers import pipeline
import torch

# https://huggingface.co/google/gemma-3-1b-it
ACCESS_TOKEN = "HUGGINGFACE_ACCESS_TOKEN_COMES_HERE"
login(ACCESS_TOKEN)

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
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
print(output[0][0]["generated_text"][2])
