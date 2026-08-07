import sys
print(sys.executable)
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Hugging Face Client
client = InferenceClient(
    api_key="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"
)

# Custom LLM Function
def llama_chat(prompt):

    prompt_text = prompt.to_string()

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content

# Convert to LangChain Runnable
llm = RunnableLambda(llama_chat)

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="""
    Generate a 5 point summary from the following text:

    {text}
    """,
    input_variables=["text"]
)

# Output Parser
parser = StrOutputParser()

# Sequential Chain
chain = (
    prompt1
    | llm
    | parser
    | prompt2
    | llm
    | parser
)

# Execute
result = chain.invoke(
    {
        "topic": "Unemployment in India"
    }
)

print("\nRESULT:\n")
print(result)

# Display Chain Graph
chain.get_graph().print_ascii()