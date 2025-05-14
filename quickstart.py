import asyncio
import json
import logging
import os
import aiohttp
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
# neo4j configs
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# LLM configs
LLM_API_KEY = "dummy"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "llama3.2:1b"

# Embedder configs
EMBEDDER_API_KEY = "dummy"
EMBEDDER_BASE_URL = "http://localhost:11434/v1"
EMBEDDER_MODEL = "nomic-embed-text"

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

"""
A custom embedder implementation for Graphiti using the sentence-transformers library
"""

import numpy as np
from typing import List, Union, Iterable
from sentence_transformers import SentenceTransformer

from graphiti_core.embedder import EmbedderClient


class SentenceTransformersEmbedder(EmbedderClient):    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize the SentenceTransformers embedder
        
        Args:
            model_name: The name of the sentence-transformers model to use
            embedding_dim: The dimension of the embeddings produced by the model
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
    
    async def create(
        self, input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> List[float]:
        """
        Create embeddings for a single input
        
        Args:
            input_data: The text to embed
            
        Returns:
            A list of floats representing the embedding
        """
        if isinstance(input_data, str):
            embedding = self.model.encode(input_data)
            return embedding.tolist()
        else:
            # Handle other input types if needed
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of inputs
        
        Args:
            input_data_list: A list of texts to embed
            
        Returns:
            A list of embeddings, each a list of floats
        """
        embeddings = self.model.encode(input_data_list)
        return embeddings.tolist()
    
class OllamaEmbedder(EmbedderClient):
    """
    An embedder client that uses the Ollama API for embeddings.
    This is tailored specifically for the Ollama API format.
    """
    
    def __init__(
        self, 
        api_key: str = "dummy", 
        base_url: str = "http://localhost:11434/v1", 
        model: str = "nomic-embed-text", 
        embedding_dim: int = 384
    ):
        """
        Initialize the Ollama embedder
        
        Args:
            api_key: API key (not usually needed for local Ollama but kept for interface consistency)
            base_url: Base URL for the Ollama API
            model: Model name to use for embeddings
            embedding_dim: The dimension of the embeddings produced by the model
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self.model = model
        self.embedding_dim = embedding_dim
    
    async def create(
        self, input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> List[float]:
        """
        Create embeddings for a single input using the Ollama API
        
        Args:
            input_data: The text to embed
            
        Returns:
            A list of floats representing the embedding
        """
        if isinstance(input_data, str):
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": input_data
                }
                
                async with session.post(f"{self.base_url}/api/embeddings", json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise ValueError(f"Ollama API error: {response.status} - {text}")
                    
                    result = await response.json()
                    # Ollama returns embeddings in a format like {"embedding": [...]}
                    if "embedding" not in result:
                        raise ValueError(f"Unexpected Ollama API response: {result}")
                    
                    return result["embedding"]
        else:
            # Handle other input types if needed
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of inputs using the Ollama API
        
        Args:
            input_data_list: A list of texts to embed
            
        Returns:
            A list of embeddings, each a list of floats
        """
        # Ollama API doesn't support batch processing directly
        # So we process each input individually
        embeddings = []
        for input_data in input_data_list:
            embedding = await self.create(input_data)
            embeddings.append(embedding)
        
        return embeddings

async def main():

    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=OpenAIGenericClient(
            config=LLMConfig(
                api_key=LLM_API_KEY,
                model=LLM_MODEL,
                base_url=LLM_BASE_URL
            )
        ),
        embedder=SentenceTransformersEmbedder(
            model_name=SENTENCE_TRANSFORMER_MODEL,
            embedding_dim=EMBEDDING_DIM
        ),
        cross_encoder=OpenAIRerankerClient(
            config=LLMConfig(
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            base_url=LLM_BASE_URL
            )
        )
    )

    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'term_start': 'January 7, 2019',
                    'term_end': 'Present',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Freakonomics Radio {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        # Print search results
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                'Who was the California Attorney General?', center_node_uuid=center_node_uuid
            )

            # Print reranked search results
            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        # Print node search results
        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
