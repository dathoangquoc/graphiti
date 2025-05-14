import asyncio
import json
import functools
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

llm_config = LLMConfig(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

embedder_config = OpenAIEmbedderConfig(
    api_key=EMBEDDER_API_KEY,
    base_url=EMBEDDER_BASE_URL,
    embedding_model=EMBEDDER_MODEL
)

def patch_openai_embedder():
    original_create_batch = OpenAIEmbedder.create_batch
    
    @functools.wraps(original_create_batch)
    async def patched_create_batch(self, texts):
        try:
            return await original_create_batch(self, texts)
        except TypeError as e:
            if "object is not iterable" in str(e):
                print("Handling Ollama embedding response format...")
                # Directly handle the embedding call ourselves
                import httpx
                
                embeddings = []
                for text in texts:
                    try:
                        response = await httpx.post(
                            f"{self.config.base_url}/embeddings",
                            json={
                                "model": self.config.embedding_model,
                                "prompt": text
                            },
                            timeout=60.0
                        )
                        
                        if response.status_code != 200:
                            print(f"Error from embedder: {response.text}")
                            raise Exception(f"Failed to get embedding: {response.status_code}")
                        
                        result = response.json()
                        if "embedding" in result:
                            embeddings.append(result["embedding"])
                        else:
                            print(f"Unexpected response format: {result}")
                            raise Exception("Embedding not found in response")
                    except Exception as inner_e:
                        print(f"Error processing embedding for text: {text[:30]}...")
                        print(f"Error details: {inner_e}")
                        raise
                
                return embeddings
            else:
                raise e
    
    # Apply the monkey patch
    OpenAIEmbedder.create_batch = patched_create_batch
    print("Applied patch to OpenAIEmbedder.create_batch")

# patch_openai_embedder()

async def main():
    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=OpenAIGenericClient(llm_config),
        embedder=OpenAIEmbedder(embedder_config),
        cross_encoder=OpenAIRerankerClient(llm_config)
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
        # for i, episode in enumerate(episodes):
        #     print('i', i)
        #     print('episode', episode)
        #     await graphiti.add_episode(
        #         name=f'Freakonomics Radio {i}',
        #         episode_body=episode['content']
        #         if isinstance(episode['content'], str)
        #         else json.dumps(episode['content']),
        #         source=episode['type'],
        #         source_description=episode['description'],
        #         reference_time=datetime.now(timezone.utc),
        #     )
        #     print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

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
