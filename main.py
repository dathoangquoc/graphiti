import asyncio
import json
import os
from datetime import datetime, timezone
import logging 

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from graphiti_core.llm_client import LLMConfig

from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient

from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from pydantic import BaseModel, Field


# neo4j configs
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# LLM configs
LLM_API_KEY = "dummy"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3:8b"

# Embedder configs
EMBEDDER_API_KEY = "dummy"
EMBEDDER_BASE_URL = "http://localhost:11434/v1"
EMBEDDER_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 384

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


# Logger
logging.basicConfig(filename=f'debug_logs/search.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
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

        # Test data
        episodes = [
            {
                'content': '岸田文雄は日本の第100代内閣総理大臣である。それ以前は外務大臣を務めていた。',
                'type': EpisodeType.text,
                'description': 'ポッドキャストの書き起こし',
            },
            {
                'content': '外務大臣としての在任期間は2012年12月26日から2017年8月3日までだった。',
                'type': EpisodeType.text,
                'description': 'ポッドキャストの書き起こし',
            },
            {
                'content': {
                    'name': '小池百合子',
                    'position': '東京都知事',
                    'previous_role': '環境大臣',
                    'previous_location': '東京',
                },
                'type': EpisodeType.json,
                'description': 'ポッドキャストのメタデータ',
            },
            {
                'content': {
                    'name': '小池百合子',
                    'position': '東京都知事',
                    'term_start': '2016年7月31日',
                    'term_end': '現在',
                },
                'type': EpisodeType.json,
                'description': 'ポッドキャストのメタデータ',
            },
        ]


        # Add episodes to the graph
        # for i, episode in enumerate(episodes):
        #     print('Adding episode:', episode)
        #     await graphiti.add_episode(
        #         group_id=f'group{i}',
        #         name=f'Freakonomics Radio {i}',
        #         episode_body=episode['content']
        #         if isinstance(episode['content'], str)
        #         else json.dumps(episode['content']),
        #         source=episode['type'],
        #         source_description=episode['description'],
        #         reference_time=datetime.now(timezone.utc),
        #         # entity_types=entity_types
        #     )
        #     print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        # Implied: 岸田文雄は長く政治に関わってきた人物である。
        query = "岸田文雄さんは総理になる前、どのような政治的経験を積んでいたのでしょうか？"
        print(f"\nSearching for: {query}")
        results = await graphiti.search(
            query=query,
            group_ids=['group1','group2','group3','group0']
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
        # if results and len(results) > 0:
        #     # Get the source node UUID from the top result
        #     center_node_uuid = results[0].source_node_uuid

        #     print('\nReranking search results based on graph distance:')
        #     print(f'Using center node UUID: {center_node_uuid}')

        #     reranked_results = await graphiti.search(
        #         query=query, center_node_uuid=center_node_uuid
        #     )

        #     # Print reranked search results
        #     print('\nReranked Search Results:')
        #     for result in reranked_results:
        #         print(f'UUID: {result.uuid}')
        #         print(f'Fact: {result.fact}')
        #         if hasattr(result, 'valid_at') and result.valid_at:
        #             print(f'Valid from: {result.valid_at}')
        #         if hasattr(result, 'invalid_at') and result.invalid_at:
        #             print(f'Valid until: {result.invalid_at}')
        #         print('---')
        # else:
        #     print('No results found in the initial search to use as center node.')

    finally:
        await graphiti.close()
        print('Connection closed')

    

if __name__ == '__main__':
    asyncio.run(main())
