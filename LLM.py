from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from openai import OpenAI
from dotenv import load_dotenv
from search import search
from rerank import rerank_chunks
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
load_dotenv()

class SearchFilters(BaseModel):
    min_experience: int = Field(None, description="Minimum years of experience required. 0 if not specified.")
    job_title_keyword: str = Field(None, description="Specific job title keyword if mentioned (e.g. 'Manager', 'Intern').")

# Initialize the LLM
llm = ChatOpenAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=os.environ["MISTRAL_API_KEY"],
    base_url="https://api.mistral.ai/v1"
)

system_template = """You are a helpful HR assistant capable of answering questions about candidate CVs.
Use the following pieces of retrieved context to answer the question.
If the answer is not in the context, say "I do not have that information." 
Do not try to make up an answer.

Context:
{context}

Question:
{question}
"""

input_query_rewriter_template = """You are a Rag Query Rewriter that rewrites the input query to be more specific and relevant to the context.
the context is CVs of candidates that have different skills and experiences and are even sometimes in different fields , 
the chunks are set up to be of different sections of each candidate email and all chunks has the metadata like the name and experience of the user,

Input Query:
{input_query}

Rewritten Query:
"""

filter_extraction_template = """
You are a search query parser. Extract search filters from the user query.
Return a JSON object with:
- "min_experience": integer (e.g., 5 for "more than 5 years", 0 if not mentioned)
- "job_title_keyword": string (e.g., "Manager" if user asks for managers, null otherwise)
User Query: {input_query}
"""


sys_prompt_template = PromptTemplate.from_template(system_template)
input_query_rewriter_template = PromptTemplate.from_template(input_query_rewriter_template)
filter_extraction_template = PromptTemplate.from_template(filter_extraction_template)

sys_prompt_chain = sys_prompt_template | llm
input_query_rewriter_chain = input_query_rewriter_template | llm
filter_chain = filter_extraction_template | llm | JsonOutputParser(pydantic_object=SearchFilters)





def extract_query_filters(query):
    try:
        return filter_chain.invoke({"input_query": query})
    except:
        return {"min_experience": 0, "job_title_keyword": None}

def format_chunk(chunk):
    return (
        f"Name: {chunk.get('full_name', 'Unknown')}\n"
        f"Job Title: {chunk.get('current_job_title', 'Unknown')}\n"
        f"Experience: {chunk.get('years_of_experience', 'Unknown')}\n"
        f"Content: {chunk.get('text_content', '')}\n"
    )


def generate_answer(context, question):
    # Join the formatted strings
    clean_context = "\n---\n".join([format_chunk(c) for c in context])
    return sys_prompt_chain.invoke({"context": clean_context, "question": question})

def run_rag_pipeline(user_query):
    # 1. Extract Filters
    filters = extract_query_filters(user_query)
    print(f"Extracted Filters: {filters}") # Debugging

    response_object = input_query_rewriter_chain.invoke({"input_query": user_query})
    rewritten_query = response_object.content

    results = search(rewritten_query, filters=filters)

    results = rerank_chunks(rewritten_query, results)

    answer = generate_answer(results, rewritten_query)
    
    return {
        "answer": answer.content,
        "contexts": [format_chunk(c) for c in results] 
    }

print(run_rag_pipeline("Give me names , current years of experience and job title of candidates that has experience with Computer Vision?"))