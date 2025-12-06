from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv
import json
import numpy as np

def embed_cvs(all_chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Your correct list comprehension
    text_chunks = [chunk["text_content"] for chunk in all_chunks]
    
    embeddings = model.encode(text_chunks)
    
    # Check the dimensions
    print(f"Shape of embeddings: {embeddings.shape}")

    np.save("embeddings.npy", embeddings)
    return embeddings
    