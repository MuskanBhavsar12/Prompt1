from huggingface_hub import InferenceClient

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser
)

from pydantic import BaseModel, Field
from typing import Literal
import json

# Hugging Face Client
client = InferenceClient(
    api_key="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"
)

# --------------------------
# Custom LLM Function
# --------------------------

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
        max_tokens=300
    )

    return response.choices[0].message.content

llm = RunnableLambda(llama_chat)

# --------------------------
# Output Parser
# --------------------------

parser = StrOutputParser()

# --------------------------
# Pydantic Schema
# --------------------------

class Feedback(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of feedback"
    )

parser2 = PydanticOutputParser(
    pydantic_object=Feedback
)

# --------------------------
# Classification Prompt
# --------------------------

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the feedback.

Feedback:
{feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction":
        parser2.get_format_instructions()
    }
)

# --------------------------
# Classifier Function
# --------------------------

def classify_feedback(inputs):

    prompt_value = prompt1.invoke(inputs)

    response = llama_chat(prompt_value)

    try:
        return parser2.parse(response)

    except Exception:

        response = response.lower()

        if "positive" in response:
            return Feedback(sentiment="positive")

        return Feedback(sentiment="negative")

classifier_chain = RunnableLambda(classify_feedback)

# --------------------------
# Positive Prompt
# --------------------------

prompt2 = PromptTemplate(
    template="""
Write an appropriate response
to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# --------------------------
# Negative Prompt
# --------------------------

prompt3 = PromptTemplate(
    template="""
Write an appropriate response
to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# --------------------------
# Branch Logic
# --------------------------

def positive_response(x):

    return (
        prompt2
        | llm
        | parser
    ).invoke(
        {
            "feedback":
            user_feedback
        }
    )


def negative_response(x):

    return (
        prompt3
        | llm
        | parser
    ).invoke(
        {
            "feedback":
            user_feedback
        }
    )

branch_chain = RunnableBranch(

    (
        lambda x: x.sentiment == "positive",
        RunnableLambda(positive_response)
    ),

    (
        lambda x: x.sentiment == "negative",
        RunnableLambda(negative_response)
    ),

    RunnableLambda(
        lambda x:
        "Could not determine sentiment"
    )
)

# --------------------------
# User Feedback
# --------------------------

user_feedback = "This is a beautiful phone"

# --------------------------
# Complete Chain
# --------------------------

chain = classifier_chain | branch_chain

# --------------------------
# Execute
# --------------------------

result = chain.invoke(
    {
        "feedback":
        user_feedback
    }
)

print("\nRESULT:\n")
print(result)

# --------------------------
# Graph
# --------------------------

chain.get_graph().print_ascii()