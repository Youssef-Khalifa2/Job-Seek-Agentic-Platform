from typing import TypedDict
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Define Constants
COLLECTION_NAME = "cv_collection"

@dataclass
class CVPayload:
    source_id: str          # Unique ID for the file (e.g., filename hash)
    candidate_name: str
    experience_years: int
    skills: List[str]
    text_content: str       # The chunk text
    chunk_index: int

# 3. The Setup Logic
def create_collection_if_not_exists(client: QdrantClient, vector_size: int = 768):
    """
    Creates the Qdrant collection with specific configuration.
    """
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,  # DEPENDS ON YOUR EMBEDDING MODEL
                distance=models.Distance.COSINE
            )
        )

        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="skills",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="text_content",
            field_schema=models.PayloadSchemaType.TEXT
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully.")
    else:
        print(f"info: Collection '{COLLECTION_NAME}' already exists.")