import asyncio
import logging
from typing import List, Optional, Dict, Any
from collections.abc import Iterable

from graphiti_core.embedder import EmbedderClient
from graphiti_core.cross_encoder import CrossEncoderClient
from sentence_transformers import SentenceTransformer, util

DEFAULT_EMBEDDING_DIM = 1024

class SentenceTransformerEmbedder(EmbedderClient):
    """Custom embedder that uses SentenceTransformer models."""
    
    def __init__(self, model_name_or_path: str, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        self.model = SentenceTransformer(model_name_or_path)
        self.embedding_dim = embedding_dim
    
    async def create(
            self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
            ) -> List[float]:
        """Create embedding for a single piece of text."""
        # Run in an executor to avoid blocking the event loop
        embedding = await asyncio.to_thread(self.model.encode, input_data, convert_to_numpy=True)
        
        # Convert numpy array to list for JSON serialization
        return embedding[:self.embedding_dim].tolist()
    
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of texts."""
        # Run in an executor to avoid blocking the event loop
        embeddings = await asyncio.to_thread(self.model.encode, input_data_list, convert_to_numpy=True)
        # Convert numpy arrays to lists for JSON serialization
        return [emb.tolist() for emb in embeddings]


class SentenceTransformerCrossEncoder(CrossEncoderClient):
    """Custom cross-encoder using SentenceTransformer for similarity scoring."""
    
    def __init__(self, model_name_or_path: str):
        self.model = SentenceTransformer(model_name_or_path)
    
    async def compute_similarity(self, query: str, passage: str) -> float:
        """Compute similarity score between query and passage."""
        # Encode both texts
        query_embedding = await asyncio.to_thread(self.model.encode, query, convert_to_numpy=True)
        passage_embedding = await asyncio.to_thread(self.model.encode, passage, convert_to_numpy=True)
        
        # Compute cosine similarity
        similarity = util.cos_sim(query_embedding, passage_embedding).item()
        return float(similarity)
    
    async def rank(self, query: str, passages: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank passages based on similarity to query."""
        # Encode query once
        query_embedding = await asyncio.to_thread(self.model.encode, query, convert_to_numpy=True)
        
        # Encode all passages
        passage_embeddings = await asyncio.to_thread(self.model.encode, passages, convert_to_numpy=True)
        
        # Compute similarities
        similarities = util.cos_sim(query_embedding, passage_embeddings)[0].tolist()
        
        # Create results with scores and indexes
        results = [{"index": i, "score": float(score)} for i, score in enumerate(similarities)]
        
        # Sort by score (descending)
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]
            
        return results