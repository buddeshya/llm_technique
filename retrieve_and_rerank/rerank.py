from typing import List, Dict
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the reranker with a cross-encoder model"""
        self.model = CrossEncoder(model_name)
    
    def rerank_documents(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of documents to return after reranking
            
        Returns:
            List of reranked document texts
        """
        # Create pairs of query and documents for scoring
        pairs = [[query, doc] for doc in documents]
        
        # Get scores for all pairs
        scores = self.model.predict(pairs)
        
        # Get indices of top-k documents
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return reranked documents
        return [documents[i] for i in top_indices] 