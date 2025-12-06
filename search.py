from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi 



model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = faiss.read_index("index.faiss")
all_chunks = json.load(open("cv_data.json", "r", encoding="utf-8"))

tokenized_corpus = [chunk['text_content'].split(" ") for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

def search(query, k=20, filters=None):
    # Default filters if None
    if filters is None:
        filters = {"min_experience": 0, "job_title_keyword": None}
    
    min_exp = filters.get("min_experience") or 0
    title_kw = filters.get("job_title_keyword")
    # Helper function to check if a document matches filters
    def is_match(doc):
        # Check Experience
        doc_exp = doc.get('years_of_experience', 0)
        # Handle cases where experience might be "Unknown" or string
        if isinstance(doc_exp, str):
             if doc_exp.isdigit(): doc_exp = int(doc_exp)
             else: doc_exp = 0
        
        if doc_exp < min_exp:
            return False
            
        # Check Job Title (Optional)
        if title_kw and title_kw.lower() not in doc.get('current_job_title', '').lower():
            return False
            
        return True
    # We fetch MORE results (k*3) to allow for filtering
    fetch_k = k * 3
    
    # 1. Vector Search
    vector_embedding = model.encode([query])
    vec_distances, vec_indices = vector_store.search(vector_embedding, k=fetch_k)
    
    # 2. BM25 Search
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[-fetch_k:][::-1]
    
    # 3. Combine Results using RRF
    rrf_constant = 60
    combined_scores = {}
    # Process Vector Results
    for rank, idx in enumerate(vec_indices[0]):
        if idx == -1: continue
        # [NEW] Apply Filter Here
        if not is_match(all_chunks[idx]):
            continue
            
        if idx not in combined_scores:
            combined_scores[idx] = 0
        combined_scores[idx] += 1 / (rank + rrf_constant)
    # Process BM25 Results
    for rank, idx in enumerate(bm25_indices):
        # [NEW] Apply Filter Here
        if not is_match(all_chunks[idx]):
            continue
        if idx not in combined_scores:
            combined_scores[idx] = 0
        combined_scores[idx] += 1 / (rank + rrf_constant)
    # 4. Sort and Format Results
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in sorted_indices[:k]:
        result_item = all_chunks[idx].copy()
        result_item['hybrid_score'] = score
        results.append(result_item)
        
    return results





    
