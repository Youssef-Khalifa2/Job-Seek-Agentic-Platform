from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
import pdfplumber
import json

load_dotenv()

client = OpenAI(
    api_key=os.environ["MISTRAL_API_KEY"],
    base_url="https://api.mistral.ai/v1"
)

class CVChunk(BaseModel):
    """Represents a single semantic section of a CV."""
    
    section_title: str = Field(..., description="The topic or header of this section (e.g., 'Work Experience', 'Skills').")
    text_content : str = Field(..., description="The text content of this section.")

   
class CVChunkList(BaseModel):
    """Container for the list of CV chunks."""
    
    chunks: List[CVChunk] = Field(..., description="A list of all the semantic sections extracted from the CV.")

class CVMetadata(BaseModel):
    """Container for the metadata of a CV."""
    
    full_name: str = Field(..., description="The full Name of the CV Owner" )
    current_job_title: str = Field(..., description="The current Job title of the CV Owner" ) 
    years_of_experience: int = Field(..., description="The years of experience of the CV Owner" )   
    skills_list: List[str] = Field(..., description="A list of skills mentioned in the CV")
    education_summary: str = Field(..., description="A summary of the education mentioned in the CV")
    key_projects: List[str] = Field(..., description="A list of key projects mentioned in the CV")
    

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts all text into a single string.
    """
    full_text = ""
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Loop through every page
        for page in pdf.pages:
            # Extract text from the current page and add a newline character
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
                
    return full_text


def extract_metadata(cv_text: str) -> CVMetadata:
    """
    Extracts metadata from a CV text using the OpenAI API.
    """
    response = client.chat.completions.create(
        model="mistral-small-latest",
        response_format={"type": "json_object"}, 
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CV Meta-Data parser. You will be given a CV text and you must output a JSON object with the following fields: "
                    "'full_name', 'current_job_title', 'years_of_experience', 'skills_list', 'education_summary', 'key_projects'. \n\n"
                    "IMPORTANT RULES:\n"
                    "- 'education_summary' must be a single string (e.g., 'Master in CS from XYZ'). Do not return an object.\n"
                    "- 'key_projects' must be a list of strings (e.g., ['Project A', 'Project B']). Do not return objects with titles/descriptions.\n"
                    "- 'years_of_experience' must be an integer if its a range take the lower number always(e.g., 2). Do not return objects with titles/descriptions.\n"

                )
            },
            {
                "role": "user", 
                "content": cv_text
            }
        ]
    )

    # Get the raw string from the LLM response
    raw_json_string = response.choices[0].message.content

    # Parse it into your Pydantic model
    cv_data = CVMetadata.model_validate_json(raw_json_string)

    # Now 'cv_data' is a real Python object, not just text!
    print(f"Successfully created Meta-data.")

    return cv_data


def extract_chunks(cv_text: str) -> CVChunkList:
    """
    Extracts chunks from a CV text using the OpenAI API.
    """
    response = client.chat.completions.create(
        model="mistral-small-latest",
        response_format={"type": "json_object"}, 
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a CV parser. You will be given a CV text and you must output a JSON object with a 'chunks' list, where each chunk has 'section_title' and 'text_content'. \n\n"
                    "IMPORTANT RULES:\n"
                    "- 'chunks' must be a list of objects with 'section_title' and 'text_content'. Do not return a single string or other objects."
                )
            },
            {
                "role": "user", 
                "content": cv_text
            }
        ]
    )

    # Get the raw string from the LLM response
    raw_json_string = response.choices[0].message.content

    # Parse it into your Pydantic model
    cv_data = CVChunkList.model_validate_json(raw_json_string)

    # Now 'cv_data' is a real Python object, not just text!
    print(f"Successfully created {len(cv_data.chunks)} chunks.")
    return cv_data


def run_llm_based_chunking(path:str):
    all_chunks = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            print(f"Starting Processing {filename}")
        cv_text = extract_text_from_pdf(os.path.join(path, filename))
        cv_metadata = extract_metadata(cv_text)
        cv_chunks = extract_chunks(cv_text)
        for cv_chunk in cv_chunks.chunks:
            all_chunks.append({
                "source_file": filename,
                "section_title": cv_chunk.section_title,
                "text_content": cv_chunk.text_content,
                **cv_metadata.model_dump()})
        if filename == "Youssef Khalifa CV.pdf":
            print(f"cv_text for youssef khalifa cv is : {cv_text}")
        print(f"Finished Processing {filename}")  
        

    output_file = "cv_data.json"

    print("Saving chunks to JSON file...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"🎉 Success! Saved {len(all_chunks)} chunks to {output_file}")

    print(f"Total chunks collected: {len(all_chunks)}")
    return all_chunks

    