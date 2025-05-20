import asyncio
from datetime import datetime, timezone
import logging 

import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Override VSCode path encoding ":" into "\x3a"

# Graphiti
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode

from graphiti_core.llm_client.openai_client import OpenAIClient

from graphiti_core.llm_client import LLMConfig

from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient

from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from docx import Document

# LangChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader


# neo4j configs
NEO4J_URI = os.environ.get('NEO4J_URI', "bolt://localhost:7687") 
NEO4J_USER = os.environ.get('NEO4J_USER', "neo4j")
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', "password")

# LLM configs
LLM_API_KEY = os.environ.get('LLM_API_KEY', "dummy")
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', "http://localhost:11434/v1")
LLM_MODEL = os.environ.get('LLM_MODEL' , "qwen3:8b")

# Embedder configs
EMBEDDER_API_KEY = os.environ.get('EMBEDDER_API_KEY', "dummy")
EMBEDDER_BASE_URL = os.environ.get("EMBEDDER_BASE_URL", "http://localhost:11434/v1")
EMBEDDER_MODEL = os.environ.get('EMBEDDER_MODEL', "nomic-embed-text")
EMBEDDING_DIM = os.environ.get('EMBEDDING_DIM', 384)

# Logger
# logging.basicConfig(filename=f'debug_logs/search.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ollama = OpenAIClient(
    config = LLMConfig(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        small_model=LLM_MODEL
    )
)

embedder = OpenAIEmbedder(
    config=OpenAIEmbedderConfig(
        embedding_model=EMBEDDER_MODEL,
        embedding_dim=EMBEDDING_DIM,
        api_key=EMBEDDER_API_KEY,
        base_url=EMBEDDER_BASE_URL
    )
)

cross_encoder = OpenAIRerankerClient(
    config = LLMConfig(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )
)

# Entity Types
class Person(BaseModel):
    name: str = Field(..., description="Full name of the person")
    role: Optional[str] = Field(None, description="The role or title of the person")
    birth_date: Optional[date] = Field(None, description="Date of birth")
    email: Optional[str] = Field(None, description="Email address")

class Place(BaseModel):
    name: str = Field(..., description="Name of the place")
    location: Optional[str] = Field(None, description="Geographic location or address")
    country: Optional[str] = Field(None, description="Country of the place")
    coordinates: Optional[tuple[float, float]] = Field(None, description="Latitude and longitude coordinates")

class Organization(BaseModel):
    name: str = Field(..., description="Name of the organization")
    type: Optional[str] = Field(None, description="Type of organization (e.g., university, company)")
    location: Optional[str] = Field(None, description="Geographic location or address")

class Event(BaseModel):
    name: str = Field(..., description="Name of the event")
    event_date: Optional[date] = Field(None, description="Date of the event")
    location: Optional[str] = Field(None, description="Location where the event takes place")

class Concept(BaseModel):
    name: str = Field(..., description="Name of the concept or term")
    definition: Optional[str] = Field(None, description="Definition or explanation")

entity_types = {
    "Person": Person,
    "Place": Place,
    "Organization": Organization,
    "Event": Event,
    "Concept": Concept,
}

def load_chunks(path: str):
    loader = DirectoryLoader(
        path=path,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
    )

    chunker = SemanticChunker(OpenAIEmbeddings(
        api_key=EMBEDDER_API_KEY,
        base_url=EMBEDDER_BASE_URL,
        model=EMBEDDER_MODEL,
        dimensions=EMBEDDING_DIM
        ), breakpoint_threshold_type="interquartile")
    
    docs = loader.lazy_load()

    for doc in docs:
        file_path = doc.metadata.get("source", "unknown")
        chunks = chunker.split_text(doc.page_content)
        yield file_path, chunks

async def add_episodes(graphiti: Graphiti, path: str):
    file_path, chunks = load_chunks(path=path)    
    
    # Add episodes to the graph
    for i, episode in enumerate(chunks):
        if isinstance(episode, str) and len(episode) > 0:
            print(f"Adding {episode}")
            await graphiti.add_episode(
                name=f'{file_path} {i}',
                episode_body=episode,
                source=EpisodeType.text,
                group_id=file_path,  # CANNOT SEARCH WITHOUT GROUP ID
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added {i}')

async def main():
    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=ollama,
        embedder=embedder,
        cross_encoder=cross_encoder
    )

    try:
        # Initialize the graph db with graphiti's indices
        await graphiti.build_indices_and_constraints()

        loader = DirectoryLoader(
        path='./data/',
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        )

        # Add docx in bulk
        for file_path, chunks in load_chunks('./data/'):
            for chunk in chunks:
                print(f'Adding {chunk}')
                await graphiti.add_episode_bulk(
                    RawEpisode(
                        name=file_path,
                        content=chunk,
                        source=EpisodeType.text,
                        reference_time=datetime.now()
                    ),
                    group_id="0"
                )

        # # Update graph
        # await graphiti.add_episode(
        #     name='wikipedia',
        #     episode_body='Hiệu ứng Magnus là hiện tượng vật lý thường liên quan đến một vật thể quay chuyển động qua chất lưu. Hiện tượng này được nhà vật lý người Đức Heinrich Gustav Magnus (1802-1870) nghiên cứu vào năm 1852.',
        #     source=EpisodeType.text,
        #     source_description='wikipedia page',
        #     group_id='wikipedia',
        #     reference_time=datetime.now()
        # )
        # print('Updated Graph')
        
        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        query = 'Sự thật về  mặt trời'
        print(f"\nSearching for: {query}")
        results = await graphiti.search(
            query=query,
            group_ids=["0"]
        )

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
        
        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            reranked_results = await graphiti.search(
                query=query, center_node_uuid=center_node_uuid
            )

            # Print reranked search results
            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

    finally:
        await graphiti.close()
        print('Connection closed')

    

if __name__ == '__main__':
    asyncio.run(main())
