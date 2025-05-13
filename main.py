import asyncio
import json
import logging
import os
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

load_dotenv()

# Connect to neo4j
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

api_key = os.environ.get('API_KEY')
base_url = os.environ.get('BASE_URL')

# Connect to LLM
llm_config = LLMConfig(
    model="llama3.2:1b",
    api_key=api_key,
    base_url=base_url
)

embedder_config = OpenAIEmbedderConfig(
    api_key=api_key,
    base_url=base_url
)

async def main():
    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=OpenAIGenericClient(llm_config),
        embedder=OpenAIEmbedder(embedder_config),
        cross_encoder=OpenAIRerankerClient(llm_config)
    )

    try:
        # Initialize the graph db with graphiti's indices
        await graphiti.build_indices_and_constraints()
    finally:
        await graphiti.close()
        print('Connection closed')

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

if __name__ == '__main__':
    asyncio.run(main())
