import numpy as np 
import faiss
import os
#from LLMBasedChunking import run_llm_based_chunking
#from embed_cvs import embed_cvs

#path_to_cvs = "CVs"

#chunks = run_llm_based_chunking(path_to_cvs)
#embeddings = embed_cvs(chunks)

vectors = np.load("embeddings.npy") # embeddings are stored in the npy file
print(vectors.shape)

index = faiss.IndexFlatIP(vectors.shape[1]) # if embeddings were in the millions we would use smth lik IndexIVFFlat as it does the clustering and could benefit the search time greatly 
index.add(vectors)
faiss.write_index(index, "index.faiss")


