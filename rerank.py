from flashrank import Ranker, RerankRequest

# ranker = Ranker() 

def rerank_chunks(query, chunks):
    print("Initializing FlashRank Ranker...")
    ranker = Ranker()
    # 1. Map our custom keys to FlashRank's expected format
    passages = []
    for i, chunk in enumerate(chunks):
        passages.append({
            "id": i,
            "text": chunk.get('text_content', ""), # Safety here too!
            "meta": chunk
        })
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    if not results:
        return []    

    top_score = results[0]['score']
    
    final_results = []
    counter = 0
    for result in results:
        # 2. Dynamic Cut: Keep if it's within a close range of the winner
        # We can also keep a safety net (e.g., always keep top 3, then filter)
        if (counter < 3) or (result['score']/top_score >= 0.8):
             final_results.append(result['meta'])
        counter += 1
            
    return final_results