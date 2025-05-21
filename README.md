# Graphiti Demo

## Purpose

The purpose of the repo is to demo and test [Graphiti](https://github.com/getzep/graphiti) abilities to index and update knowledge graphs. 

## How to run

1. Start Neo4j DB

```bash
bash start_neo4j.sh
```

2. Create your .env with the configs at the top of main.py

3. Run the app

```bash
python main.py
```

## Managing DB

Go to http://localhost:7474/ to manage the DB

# Benchmark
## Indexing & Updating
- Average: 22 min/1000 words ~ 1.3s/word
- The long processing time is due to how episodes are processed. LLM inference speed is the main bottleneck. 
```
for each episode:
	call_llm('get entities')
	call_llm('get facts')
	
	# Deduplicates 
	call_llm('dedupe entities from list')
	call_llm('dedupe facts from list')
	
	for each entity:
		call_llm('get attributes')
	
	for each fact:
		call_llm('get contradicted facts')
```

## Query
- Average: 0.1s/query
- Accuracy: 
	- Search requires graph partition (specifying group_id), affects accuracy.
	- Top result is often the node with more edges than the correct entity node. Correct entity node is found within the top 3

# Possible improvements
1. Improve indexing speed
- Calling `add_episode` in parallel for different `group_id` 
- Adding max number of edge and node in prompt
- Reduce or remove [reflexion process](https://arxiv.org/abs/2303.11366)
