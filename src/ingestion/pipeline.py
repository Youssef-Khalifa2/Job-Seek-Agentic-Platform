import numpy as np
import os
import uuid  # <--- WAS MISSING
from pydantic import BaseModel, Field
from typing import List
import config
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from src.database.schema import COLLECTION_NAME
from qdrant_client.http.models import PointStruct
from src.database.client import VectorDBClient

qdrant_client = VectorDBClient()

# Initialize Mistral Client
client = OpenAI(
    api_key=config.MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

# --- Data Models ---
class CVChunk(BaseModel):
    section_title: str = Field(..., description="The topic or header (e.g., 'Work Experience').")
    text_content : str = Field(..., description="The text content of this section.")

class CVChunkList(BaseModel):
    chunks: List[CVChunk] = Field(..., description="List of semantic sections.")

class CVMetadata(BaseModel):
    full_name: str = Field(..., description="The full Name of the CV Owner" )
    current_job_title: str = Field(..., description="The current Job title" ) 
    years_of_experience: int = Field(..., description="Years of experience (integer)" )   
    skills_list: List[str] = Field(..., description="List of skills")
    education_summary: str = Field(..., description="Summary of education")
    key_projects: List[str] = Field(..., description="List of key projects")

# --- Extraction Functions ---
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

def extract_metadata(cv_text: str) -> CVMetadata:
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
    return CVMetadata.model_validate_json(response.choices[0].message.content)

def extract_chunks(cv_text: str) -> CVChunkList:
    """
    Extracts semantic chunks from CV text, ensuring NO details are lost.
    """
    response = client.chat.completions.create(
        model="mistral-small-latest",
        response_format={"type": "json_object"}, 
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are a CV Segmentation Engine. Your job is to split the CV text into semantic sections (e.g., 'Work Experience', 'Projects', 'Education').\n\n"
                    "CRITICAL RULES:\n"
                    "1. For 'text_content', you must copy the FULL original text for that section, including ALL bullet points, descriptions, metrics, and dates.\n"
                    "2. DO NOT summarize. DO NOT truncate. DO NOT exclude any lines.\n"
                    "3. Return a JSON object with a 'chunks' list containing 'section_title' and 'text_content'."
                )
            },
            {
                "role": "user", 
                "content": cv_text
            }
        ]
    )
    
    # Parse the response
    return CVChunkList.model_validate_json(response.choices[0].message.content)

# --- Core Logic ---
def run_llm_based_chunking(path: str):
    all_chunks = []
    filename = os.path.basename(path) # <--- Corrected scope

    cv_text = extract_text_from_pdf(path)
    print(f"📄 Extracted text from {filename}")

    cv_metadata = extract_metadata(cv_text)
    print("🧠 Metadata extracted")

    cv_chunks = extract_chunks(cv_text)
    print(f"🔪 Split into {len(cv_chunks.chunks)} chunks")

    for i, cv_chunk in enumerate(cv_chunks.chunks):
        all_chunks.append({
            "source_id": filename,
            "chunk_index": i,
            "section_title": cv_chunk.section_title,
            "text_content": cv_chunk.text_content,
            # Flatten metadata
            "candidate_name": cv_metadata.full_name,
            "experience_years": cv_metadata.years_of_experience,
            "skills": cv_metadata.skills_list
        })
    
    return all_chunks

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=config.GEMINI_API_KEY
    )

def embed_cvs(texts):
    """
    Expects a list of STRINGS, not dicts.
    """
    model = get_embedding_model()
    embeddings = model.embed_documents(texts)
    # Convert to numpy only for shape check, return list for Qdrant
    print(f"Shape of embeddings: {np.array(embeddings).shape}")
    return embeddings

def ingest_cv(pdf_file_path: str):
    """
    Parses a PDF, chunks the text, embeds it, and upserts to the Vector DB.
    """
    # 1. Processing
    chunks = run_llm_based_chunking(pdf_file_path) # <--- Variable name is 'chunks'

    # 2. Embedding
    texts_to_embed = [chunk["text_content"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
    
    embeddings = embed_cvs(texts_to_embed) # <--- Passed list of strings correctly

    # 3. Packaging for Qdrant
    points = []
    for i, chunk_dict in enumerate(chunks): # <--- Fixed: used 'chunks', not 'chunks_data'
        point = PointStruct(
            id=str(uuid.uuid4()), 
            vector=embeddings[i], 
            payload=chunk_dict
        )
        points.append(point)
    
    # Use the method you just wrote
    operation_info = qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"✅ Upserted {len(points)} chunks. Status: {operation_info.status}")