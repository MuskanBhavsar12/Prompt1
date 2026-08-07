from huggingface_hub import InferenceClient
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Hugging Face Client
client = InferenceClient(
    api_key="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"
)

# Messages
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about LangChain")
]

# Convert LangChain Messages to Hugging Face Format
hf_messages = []

for msg in messages:

    if isinstance(msg, SystemMessage):
        role = "system"

    elif isinstance(msg, HumanMessage):
        role = "user"

    else:
        role = "assistant"

    hf_messages.append(
        {
            "role": role,
            "content": msg.content
        }
    )

# Generate Response
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=hf_messages,
    max_tokens=300
)

# Store AI Response
messages.append(
    AIMessage(
        content=response.choices[0].message.content
    )
)

# Print Complete Conversation
print(messages)