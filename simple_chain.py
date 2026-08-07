from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Hugging Face Client
client = InferenceClient(
    api_key="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"
)

# Function for Hugging Face
def llama_chat(prompt):

    # Convert PromptValue to string
    prompt_text = prompt.to_string()

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content

# LangChain Wrapper
llm = RunnableLambda(llama_chat)

# Prompt Template
prompt = PromptTemplate(
    template="Explain {topic} in simple words with examples.",
    input_variables=["topic"]
)

# Output Parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

# Test API
print("Testing API...")

test_response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

print(test_response.choices[0].message.content)

# Execute Chain
result = chain.invoke(
    {
        "topic": "Artificial Intelligence"
    }
)

print("\nResponse:\n")
print(result)