# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # chat template
# chat_template = ChatPromptTemplate([
#     ('system','You are a helpful customer support agent'),
#     MessagesPlaceholder(variable_name='chat_history'),
#     ('human','{query}')
# ])

# chat_history = []
# # load chat history
# with open('chat_history.txt') as f:
#     chat_history.extend(f.readlines())

# print(chat_history)

# # create prompt
# prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

# print(prompt)






from huggingface_hub import InferenceClient
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

client = InferenceClient(
    api_key="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"
)

# Chat Template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Load Chat History
chat_history = []

with open("1chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

# Generate Prompt
prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "query": "Where is my refund?"
    }
)

# Send to Llama
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": prompt.to_string()
        }
    ]
)

print("\nAI Response:\n")
print(response.choices[0].message.content)