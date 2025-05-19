import asyncio
from datetime import datetime, timezone
import logging 

import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Override VSCode path encoding ":" into "\x3a"

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
from docx import Document


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

# # Entity Types
# class Person(BaseModel):
#     role: str | None = Field(..., description="The role of the person")

# class Place(BaseModel):
#     location: str | None = Field(..., description="The location of the place")

# entity_types = {"Person": Person, "Place": Place}

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

        doc_name = 'Những kiến thức khoa học kỳ thú, không có trong sách vở'
        doc = Document(f'data/{doc_name}.docx')

        # Add episodes to the graph
        # for i, episode in enumerate(doc.paragraphs):
        #     if isinstance(episode.text, str) and len(episode.text) > 0:
        #         print(f"Adding {episode.text}")
        #         await graphiti.add_episode(
        #             name=f'{doc_name} {i}',
        #             episode_body=episode.text,
        #             source=EpisodeType.text,
        #             group_id=doc_name,  # CANNOT SEARCH WITHOUT GROUP ID
        #             source_description='article about science',
        #             reference_time=datetime.now(timezone.utc),
        #         )
        #         print(f'Added paragraph {i}')
        
        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        query = 'Magnus'
        print(f"\nSearching for: {query}")
        results = await graphiti.search(
            query=query,
            group_ids=[doc_name]
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
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                query=query, center_node_uuid=center_node_uuid
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

    finally:
        await graphiti.close()
        print('Connection closed')

    

if __name__ == '__main__':
    asyncio.run(main())
