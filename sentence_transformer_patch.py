import asyncio
import numpy as np
from typing import List, Optional, Union, Dict, Any

from graphiti_core.embedder import EmbedderClient
from graphiti_core.cross_encoder import CrossEncoderClient
from sentence_transformers import SentenceTransformer, util

class SentenceTransformerEmbedder(EmbedderClient):
    """Custom embedder that uses SentenceTransformer models."""
    
    def __init__(self, model_name_or_path: str, embedding_dimension: int = None):
        self.model = SentenceTransformer(model_name_or_path)
        self.embedding_dimension = embedding_dimension
    
    async def create(self, text: str = None, input_data: str = None, **kwargs) -> List[float]:
        """Create embedding for a single piece of text.
        
        Args:
            text (str, optional): Input text to embed
            input_data (str, optional): Alternative input text (used by Graphiti)
            **kwargs: Any additional arguments
            
        Returns:
            List[float]: The embedding vector
        """
        # Use either text or input_data, whichever is provided
        input_text = input_data if input_data is not None else text
        if input_text is None:
            raise ValueError("Either 'text' or 'input_data' must be provided")
        
        print(f"Creating embedding for: {input_text[:30]}...")
        
        # Run in an executor to avoid blocking the event loop
        embedding = await asyncio.to_thread(self.model.encode, input_text, convert_to_numpy=True)
        
        # Make sure embedding has the correct dimension
        if len(embedding) != self.embedding_dimension:
            print(f"Warning: Embedding dimension mismatch. Got {len(embedding)}, expected {self.embedding_dimension}")
            # Pad or truncate if needed
            if len(embedding) < self.embedding_dimension:
                embedding = np.pad(embedding, (0, self.embedding_dimension - len(embedding)))
            else:
                embedding = embedding[:self.embedding_dimension]
        
        # Convert numpy array to list for JSON serialization
        return embedding.tolist()
    
    async def create_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts."""
        if not texts:
            return []
            
        print(f"Creating batch embeddings for {len(texts)} texts")
        
        # Run in an executor to avoid blocking the event loop
        embeddings = await asyncio.to_thread(self.model.encode, texts, convert_to_numpy=True)
        
        # Convert numpy arrays to lists for JSON serialization
        result = [emb.tolist() for emb in embeddings]
        
        # Ensure all embeddings have the correct dimension
        for i, emb in enumerate(result):
            if len(emb) != self.embedding_dimension:
                print(f"Warning: Embedding {i} dimension mismatch. Got {len(emb)}, expected {self.embedding_dimension}")
                # Pad or truncate if needed
                if len(emb) < self.embedding_dimension:
                    result[i] = emb + [0.0] * (self.embedding_dimension - len(emb))
                else:
                    result[i] = emb[:self.embedding_dimension]
        
        return result

class SentenceTransformerCrossEncoder(CrossEncoderClient):
    """Custom cross-encoder using SentenceTransformer for similarity scoring."""
    
    def __init__(self, model_name_or_path: str):
        self.model = SentenceTransformer(model_name_or_path)
    
    def __init__(self, model_name_or_path: str):
        self.model = SentenceTransformer(model_name_or_path)
        print(f"SentenceTransformerCrossEncoder initialized with model: {model_name_or_path}")
    
    async def compute_similarity(self, query: str, passage: str) -> float:
        """Compute similarity score between query and passage."""
        try:
            print(f"Computing similarity between: '{query[:30]}...' and '{passage[:30]}...'")
            
            # Encode both texts
            query_embedding = await asyncio.to_thread(self.model.encode, query, convert_to_numpy=True)
            passage_embedding = await asyncio.to_thread(self.model.encode, passage, convert_to_numpy=True)
            
            # Compute cosine similarity
            similarity = util.cos_sim(query_embedding, passage_embedding).item()
            print(f"Similarity score: {similarity}")
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            # Return a default similarity of 0.0 in case of errors
            return 0.0
    
    async def rank(self, query: str, passages: List[str], top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """Rerank passages based on similarity to query.
        
        Args:
            query (str): The query text
            passages (List[str]): List of passages to rank
            top_k (Optional[int]): Number of top results to return
            **kwargs: Additional parameters that might be passed
            
        Returns:
            List[Dict[str, Any]]: Ranked results with scores
        """
        if not passages:
            return []
            
        try:
            print(f"Reranking {len(passages)} passages for query: '{query[:30]}...'")
            
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
                
            print(f"Reranking complete. Top score: {results[0]['score'] if results else 'N/A'}")
            return results
        except Exception as e:
            print(f"Error during reranking: {str(e)}")
            # Return empty results on error
            return []
