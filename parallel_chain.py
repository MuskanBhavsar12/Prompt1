from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# HuggingFace Client
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
        max_tokens=500
    )

    return response.choices[0].message.content


# Convert Function to LangChain Runnable
llm = RunnableLambda(llama_chat)

# Prompt 1
prompt1 = PromptTemplate(
    template="""
    Generate short and simple notes from the following text:

    {text}
    """,
    input_variables=["text"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="""
    Generate 5 short question answers from the following text:

    {text}
    """,
    input_variables=["text"]
)

# Prompt 3
prompt3 = PromptTemplate(
    template="""
    Merge the provided notes and quiz into a single document.

    Notes:
    {notes}

    Quiz:
    {quiz}
    """,
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Parallel Execution
parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | llm | parser,
        "quiz": prompt2 | llm | parser
    }
)

# Merge Step
merge_chain = prompt3 | llm | parser

# Complete Chain
chain = parallel_chain | merge_chain

# Input Text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions.

SVMs do not directly provide probability estimates.

The support vector machines in scikit-learn support both dense and sparse sample vectors as input.
"""

# Execute
result = chain.invoke(
    {
        "text": text
    }
)

print("\nRESULT:\n")
print(result)

# View Graph
chain.get_graph().print_ascii()