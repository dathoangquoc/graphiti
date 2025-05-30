curl http://216.98.85.189:3641/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [
      {
        "role": "system",
        "content": "You are an AI assistant that extracts entity nodes from text. \nYour primary task is to extract and classify the speaker and other significant entities mentioned in the provided text.\nDo not escape unicode characters.\nAny extracted information should be returned in the same language as it was written in."
      },
      {
        "role": "user",
        "content": "\n<TEXT>\nKamala Harris is the Attorney General of California. She was previously the district attorney for San Francisco.\n</TEXT>\n<ENTITY TYPES>\n[{\"entity_type_id\": 0, \"entity_type_name\": \"Entity\", \"entity_type_description\": \"Default entity classification. Use this entity type if the entity is not one of the other listed types.\"}, {\"entity_type_id\": 1, \"entity_type_name\": \"Person\", \"entity_type_description\": null}, {\"entity_type_id\": 2, \"entity_type_name\": \"Place\", \"entity_type_description\": null}]\n</ENTITY TYPES>\n\nGiven the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.\nFor each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.\nIndicate the classified entity type by providing its entity_type_id.\n\nGuidelines:\n1. Extract significant entities, concepts, or actors mentioned in the conversation.\n2. Avoid creating nodes for relationships or actions.\n3. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).\n4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.\n\nRespond with a JSON object in the following format:\n\n{\"$defs\": {\"ExtractedEntity\": {\"properties\": {\"name\": {\"description\": \"Name of the extracted entity\", \"title\": \"Name\", \"type\": \"string\"}, \"entity_type_id\": {\"description\": \"ID of the classified entity type. Must be one of the provided entity_type_id integers.\", \"title\": \"Entity Type Id\", \"type\": \"integer\"}}, \"required\": [\"name\", \"entity_type_id\"], \"title\": \"ExtractedEntity\", \"type\": \"object\"}}, \"properties\": {\"extracted_entities\": {\"description\": \"List of extracted entities\", \"items\": {\"$ref\": \"#/$defs/ExtractedEntity\"}, \"title\": \"Extracted Entities\", \"type\": \"array\"}}, \"required\": [\"extracted_entities\"], \"title\": \"ExtractedEntities\", \"type\": \"object\"}"
      }
    ]
  }'
