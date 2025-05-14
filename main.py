import asyncio
import json
from datetime import datetime, timezone
from logging import INFO

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client import LLMConfig

from sentence_transformer_patch import SentenceTransformerEmbedder, SentenceTransformerCrossEncoder

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


llm_config = LLMConfig(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

embedder_config = LLMConfig(
    model=EMBEDDER_MODEL,
    api_key=EMBEDDER_API_KEY,
    base_url=EMBEDDER_BASE_URL,
)

async def main():
    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=OpenAIGenericClient(llm_config),
        embedder=SentenceTransformerEmbedder(SENTENCE_TRANSFORMER_MODEL, EMBEDDING_DIM),
        cross_encoder=SentenceTransformerCrossEncoder(SENTENCE_TRANSFORMER_MODEL)
    )

    try:
        # Initialize the graph db with graphiti's indices
        await graphiti.build_indices_and_constraints()

        # Test data
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
            print('Adding episode:', episode)
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

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        query = "Who is Kamala Harris?"
        print(f"\nSearching for: {query}")
        results = await graphiti.search(query)

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

    finally:
        await graphiti.close()
        print('Connection closed')

    

if __name__ == '__main__':
    asyncio.run(main())
