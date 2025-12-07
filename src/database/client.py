import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VectorDBClient:
    """
    Manages connections to the Vector Database (Qdrant).
    """
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        
        self.client = QdrantClient(host=host, port=port)
        print(f"🔌 Connected to Qdrant at {host}:{port}")

    def upsert(self, collection_name, points):
        """
        Uploads (inserts or updates) points to the database.
        Args:
            collection_name (str): Name of the collection.
            points (list): List of PointStruct objects.
        """
        operation_info = self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True  # Important: Waits for the database to confirm storage
        )
        return operation_info

    def search(self, collection_name, vector, limit=3, filter_conditions=None):
        """
        Searches for the nearest vectors.
        Args:
            collection_name (str): Name of the collection.
            vector (list): The query embedding (list of floats).
            limit (int): How many results to return.
            filter_conditions (dict, optional): Qdrant filter object for metadata filtering.
        """
        results = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            query_filter=filter_conditions
        ).points
        return results

# Helper to easily get an instance anywhere
def get_client():
    return VectorDBClient()