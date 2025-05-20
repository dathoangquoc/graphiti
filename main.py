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
from docx import Document

# LangChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


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
    role: Optional[str] = Field(None, description="The role or title of the person")

class Place(BaseModel):
    location: Optional[str] = Field(None, description="Geographic location or address")
    country: Optional[str] = Field(None, description="Country of the place")
    coordinates: Optional[tuple[float, float]] = Field(None, description="Latitude and longitude coordinates")

class Organization(BaseModel):
    type: Optional[str] = Field(None, description="Type of organization (e.g., university, company)")
    location: Optional[str] = Field(None, description="Geographic location or address")

class Event(BaseModel):
    date: Optional[str] = Field(None, description="Date of the event")
    location: Optional[str] = Field(None, description="Location where the event takes place")

class Concept(BaseModel):
    definition: Optional[str] = Field(None, description="Definition or explanation")

entity_types = {
    "Person": Person,
    "Place": Place,
    "Organization": Organization,
    "Event": Event,
    "Concept": Concept,
}

def load_docx_files_from_dir(directory: str):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".docx"):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)
                print(f"Loading file: {file_path}")
                doc = Document(file_path)
                
                content = []
                for para in doc.paragraphs:
                    text = para.text.strip()
                    if isinstance(text, str) and len(text) > 1:
                        content.append(text)
                
                yield file_name, content

def load_chunks(path: str):
    chunker = SemanticChunker(OllamaEmbeddings(
        base_url='http://localhost:11434/',
        model=EMBEDDER_MODEL),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
        min_chunk_size=5
        )
    
    for file_name, content in load_docx_files_from_dir(path):
        chunks = chunker.create_documents(content)
        yield file_name, chunks

async def add_episodes(graphiti: Graphiti, path: str):
    for file_name, chunks in load_chunks(path=path):        
        # Add episodes to the graph
        for i, episode in enumerate(chunks):
            episode_text = episode.page_content
            if isinstance(episode_text, str) and len(episode_text) > 0:
                print(f"Adding {episode}")
                await graphiti.add_episode(
                    name=f'{file_name} {i}',
                    episode_body=episode_text,
                    source=EpisodeType.text,
                    source_description='Word Document',
                    group_id=file_name,  # CANNOT SEARCH WITHOUT GROUP ID
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types
                )
                print(f'Added {i}')
            else:
                print('Error adding episode: ', episode)


async def add_episodes_bulk(graphiti: Graphiti, path: str):
    """Process multiple episodes from documents in bulk."""
    try:
        # Collect all episodes in a list
        all_episodes = []
        
        # Track document sources for debugging
        source_counts = {}
        
        for file_name, chunks in load_chunks(path=path):
            source_counts[file_name] = len(chunks)
            print(f"Found {len(chunks)} chunks in {file_name}")
            
            for i, chunk in enumerate(chunks):
                # Extract the text from the Document object
                if hasattr(chunk, 'page_content'):
                    episode_text = chunk.page_content
                else:
                    # Fallback approach if page_content doesn't exist
                    episode_text = str(chunk)
                
                if episode_text and len(episode_text) > 0:
                    # Create a RawEpisode object
                    episode = RawEpisode(
                        name=f'{file_name}_{i}',
                        source=EpisodeType.text,
                        content=episode_text,
                        source_description='Word Document',
                        reference_time=datetime.now(timezone.utc)
                    )
                    all_episodes.append(episode)
        
        if not all_episodes:
            print("No valid episodes found to add.")
            return
        
        print(f"Preparing to add {len(all_episodes)} episodes in bulk")
        print(f"Document sources: {source_counts}")
        
        # Process all episodes in bulk
        group_id = '0'
        await graphiti.add_episode_bulk(all_episodes, group_id=group_id)
        
        print(f"Successfully added {len(all_episodes)} episodes in bulk with group_id: {group_id}")
    
    except Exception as e:
        print(f"Error in bulk episode processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    

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

        # Add episodes in bulk
        # await add_episodes_bulk(graphiti, './data/')  # NON FUNCTIONAL

        # Add each episode
        await add_episodes(graphiti, './data/')

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
