import datetime
import asyncio
import re
from typing import List, Optional, Dict, Any, Iterable
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from ..clients import (
    extraction_model, 
    embedding_model, 
    neo4j_gemini_model,
    get_postgres_connection,
    get_neo4j_connection
)
import json, string

# --- [SETTINGS] ---
# 60 seconds / 10 requests = 6s per request. We add a buffer.
THROTTLE_WAIT_SECONDS = 6.1

PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)

def normalize_string(text: str) -> str:
    """Removes punctuation, newlines, and extra whitespace for matching."""
    if not text:
        return ""
    # 1. Remove punctuation
    text_no_punct = text.translate(PUNCTUATION_TRANSLATOR)
    # 2. Lowercase and split into words (this handles newlines)
    words = text_no_punct.lower().split()
    # 3. Re-join with single spaces
    return " ".join(words)


# --- 1. Pydantic Models ---
class Event(BaseModel):
    event_name: str = Field(description="The name of the event")
    people_involved: List[str] = Field(description="List of people involved in the event")
    date: str = Field(description="Date of the event in YYYY-MM-DD format. Use today's date if not specified.")
    location: Optional[str] = Field(description="Location of the event, or empty string", default="")
    source_sentence: str = Field(description="The exact sentence from the transcript that contains this information")

class Action(BaseModel):
    action_name: str = Field(description="The name of the action item")
    people_involved: List[str] = Field(description="List of people assigned to the action")
    date: str = Field(description="Due date of the action in YYYY-MM-DD format. Use today's date if not specified.")
    location: Optional[str] = Field(description="Location, or empty string", default="")
    source_sentence: str = Field(description="The exact sentence from the transcript that contains this information")

class TranscriptData(BaseModel):
    events: List[Event] = Field(description="A list of all extracted events")
    actions: List[Action] = Field(description="A list of all extracted action items")

class ParsedLine(BaseModel):
    line_id: int
    start_time: float
    end_time: float
    speaker: str
    text: str
    speaker_entity_id: int
    spokentime: datetime.datetime
    normalized_text: str  # New field for normalized text

class Neo4jWriter:
    """
    A class to handle all write operations to the Neo4j database.
    It uses transaction functions for data safety.
    """
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Neo4j connection successful.")
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            print("Please ensure Neo4j is running and credentials are correct.")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def populate_graph_from_data(self, graph_data):
        """
        Main function to populate the graph from the LLM's JSON output.
        """
        if not self.driver:
            print("Cannot populate graph, Neo4j driver is not initialized.")
            return

        with self.driver.session() as session:
            
            # 1. Process Interactions
            for item in graph_data.get('interactions', []):
                if item.get('person1') and item.get('person2'):
                    session.execute_write(
                        self._create_interaction, 
                        item['person1'], 
                        item['person2']
                    )

            # 2. Process Preferences
            for item in graph_data.get('preferences', []):
                if item.get('person') and item.get('item') and item.get('type'):
                    session.execute_write(
                        self._create_preference,
                        item['person'],
                        item['item'],
                        item['type']  # 'LIKES' or 'DISLIKES'
                    )

            # 3. Process Opinions
            for item in graph_data.get('opinions', []):
                # --- VALIDATION ---
                if item.get('person') and item.get('target_name') and item.get('target_type'):
                    session.execute_write(
                        self._create_opinion,
                        item['person'],
                        item['target_name'],
                        item['target_type'],  # 'Topic' or 'Person'
                        item.get('sentiment', 'neutral'),
                        item.get('quote', '')
                    )
            
            # 4. Process Emotions
            for item in graph_data.get('emotions', []):
                if item.get('person') and item.get('emotion'):
                    session.execute_write(
                        self._create_emotion,
                        item['person'],
                        item['emotion'],
                        item.get('topic', 'general')
                    )

            # 5. Process Intents
            for item in graph_data.get('intents', []):
                if item.get('person') and item.get('intent'):
                    session.execute_write(
                        self._create_intent,
                        item['person'],
                        item['intent'],
                        item.get('target', 'unknown'),
                        item.get('statement', '')  # <-- ADD THIS LINE
                    )

    # --- Static methods for Cypher queries (run within a transaction) ---
    
    @staticmethod
    def _create_interaction(tx, person1_name, person2_name):
        # We model both KNOWS and a weighted INTERACTS_WITH
        query = """
        MERGE (p1:Person {name: $person1_name})
        MERGE (p2:Person {name: $person2_name})
        MERGE (p1)-[:KNOWS]->(p2)
        MERGE (p1)-[r:INTERACTS_WITH]->(p2)
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
        """
        tx.run(query, person1_name=person1_name, person2_name=person2_name)

    @staticmethod
    def _create_preference(tx, person_name, item_name, pref_type):
        query = """
        MERGE (p:Person {name: $person_name})
        MERGE (i:Item {name: $item_name})
        """
        # Use separate MERGE statements for LIKES/DISLIKES
        if pref_type.upper() == 'LIKES':
            query += " MERGE (p)-[:LIKES]->(i)"
        elif pref_type.upper() == 'DISLIKES':
            query += " MERGE (p)-[:DISLIKES]->(i)"
        
        tx.run(query, person_name=person_name, item_name=item_name)

    @staticmethod
    def _create_opinion(tx, person_name, target_name, target_type, sentiment, quote):
        # This query dynamically merges the target node based on its type
        # NOTE: This requires the APOC plugin for 'apoc.merge.node'
        # If APOC isn't installed, this query will fail.
        # A simpler version would be to label all targets as ':Topic'
        query = """
        MERGE (p:Person {name: $person_name})
        WITH p
        CALL apoc.merge.node([$target_type], {name: $target_name}) YIELD node as t
        MERGE (p)-[r:HAS_OPINION]->(t)
            SET r.sentiment = $sentiment, r.quote = $quote
        """
        tx.run(query, person_name=person_name, target_name=target_name, 
               target_type=target_type.capitalize(), # 'Topic' or 'Person'
               sentiment=sentiment, quote=quote)

    @staticmethod
    def _create_emotion(tx, person_name, emotion_type, topic):
        query = """
        MERGE (p:Person {name: $person_name})
        MERGE (e:Emotion {type: $emotion_type, topic: $topic})
        MERGE (p)-[:FEELS]->(e)
        """
        tx.run(query, person_name=person_name, emotion_type=emotion_type, topic=topic)

    @staticmethod
    def _create_intent(tx, person_name, intent_type, target, statement):
        query = """
        MERGE (p:Person {name: $person_name})
        MERGE (g:Goal {type: $intent_type, target: $target})
        MERGE (p)-[r:HAD_INTENT]->(g)
            SET r.statement = $statement 
        """
        tx.run(query, person_name=person_name, intent_type=intent_type, 
               target=target, statement=statement)

# --- 2. Database Helper Tools ---
def get_or_create_id(cursor, table: str, lookup_col: str, lookup_val: str, return_col: str):
    sql_find = f"SELECT {return_col} FROM {table} WHERE {lookup_col} = %s"
    cursor.execute(sql_find, (lookup_val,))
    row = cursor.fetchone()
    if row:
        return row[0]
    sql_insert = f"INSERT INTO {table} ({lookup_col}) VALUES (%s) RETURNING {return_col}"
    cursor.execute(sql_insert, (lookup_val,))
    return cursor.fetchone()[0]

def get_or_create_location(cursor, city_name: str) -> int:
    if not city_name or not city_name.strip():
        return None
    sql_find = "SELECT location_id FROM locations WHERE city = %s"
    cursor.execute(sql_find, (city_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    sql_insert = "INSERT INTO locations (country, city) VALUES (%s, %s) RETURNING location_id"
    cursor.execute(sql_insert, (city_name, city_name))
    return cursor.fetchone()[0]

def parse_transcript_and_populate_lines(transcript: str) -> List[ParsedLine]:
    print("--- [Tool] Parsing transcript and populating 'lines' table... ---")
    line_pattern = re.compile(
        r"\[(\d+\.\d+)-(\d+\.\d+)\] "
        r"(.*?): "
        r"(.*?)(?=\n\[\d|\Z)", 
        re.DOTALL
    )
    parsed_lines = []
    today = datetime.date.today()
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cursor:
            for match in line_pattern.finditer(transcript):
                start, end, speaker, text = match.groups()
                start_f = float(start)
                text = text.strip()
                
                speaker_entity_id = get_or_create_id(
                    cursor, "entities", "entity_name", speaker.strip(), "entity_id"
                )
                spokentime = datetime.datetime(
                    today.year, today.month, today.day, 
                    tzinfo=datetime.timezone.utc
                ) + datetime.timedelta(seconds=start_f)
                
                cursor.execute(
                    """
                    INSERT INTO lines (line_text, spokentime, entity_id) 
                    VALUES (%s, %s, %s) RETURNING line_id
                    """,
                    (text, spokentime, speaker_entity_id)
                )
                line_id = cursor.fetchone()[0]
                
                # We create the ParsedLine object WITH the normalized text
                parsed_lines.append(ParsedLine(
                    line_id=line_id, start_time=start_f, end_time=float(end),
                    speaker=speaker.strip(), text=text,
                    speaker_entity_id=speaker_entity_id, spokentime=spokentime,
                    normalized_text=normalize_string(text) # Store normalized version
                ))
            conn.commit()
            print(f"--- [Tool] Successfully inserted {len(parsed_lines)} lines into DB. ---")
            return parsed_lines
    except Exception as e:
        conn.rollback()
        print(f"Error in parse_transcript_and_populate_lines: {e}")
        return []
    finally:
        conn.close()

def group_lines_for_summarization(lines: List[ParsedLine], lines_per_group: int = 10, overlap: int = 2) -> List[Dict[str, Any]]:
    print(f"--- [Tool] Grouping {len(lines)} lines into groups... ---")
    grouped_for_summary = []
    
    for i in range(0, len(lines), lines_per_group - overlap):
        group = lines[i : i + lines_per_group]
        if not group:
            continue
            
        full_text_of_group = "\n".join([f"[{line.speaker}]: {line.text}" for line in group])
        line_ids_in_group = [line.line_id for line in group]
        
        grouped_for_summary.append({
            "text_to_summarize": full_text_of_group,
            "line_ids": line_ids_in_group
        })
    print(f"--- [Tool] Created {len(grouped_for_summary)} summary groups. ---")
    return grouped_for_summary

async def summarize_line_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print(f"--- [Tool] Serially summarizing {len(groups)} groups of lines... ---")

    async def summarize_one_group(group: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
            You are a helpful assistant that summarizes conversations...
            Transcript Part:
            {group['text_to_summarize']}
        """
        try:
            # --- [THROTTLE] Wait before summary (chat) call ---
            print(f"--- [Throttling] Waiting {THROTTLE_WAIT_SECONDS}s before summary... ---")
            await asyncio.sleep(THROTTLE_WAIT_SECONDS)
            response = await extraction_model.ainvoke(prompt)
            summary_text = response.content.strip()
            
            # --- [OPTIMIZATION] No wait before embedding call.
            print("--- [Embedding] Calling embedding model (no wait)... ---")
            summary_embedding = await embedding_model.aembed_query(summary_text)
            
            return {
                "summary_text": summary_text,
                "summary_embedding": summary_embedding,
                "line_ids": group['line_ids']
            }
        except Exception as e:
            print(f"Error summarizing group: {e}")
            return {
                "summary_text": "Error", 
                "summary_embedding": None, 
                "line_ids": group['line_ids']
            }

    summaries = []
    for i, group in enumerate(groups):
        print(f"--- [Throttling] Processing summary group {i + 1} of {len(groups)} ---")
        summary_obj = await summarize_one_group(group)
        summaries.append(summary_obj)
            
    return summaries

async def extract_info_json(transcript: str) -> dict:
    print("--- [Tool] Extracting JSON info from full transcript... ---")
    current_date = datetime.date.today().isoformat()
    prompt = f"""
    You are a useful information extraction system.
    Extract all events and action items from the transcript.
    The current date is: {current_date}. Use this if no other date is specified.
    
    Transcript:
    "{transcript}"
    """
    try:
        # --- [THROTTLE] Wait before extraction (chat) call ---
        print(f"--- [Throttling] Waiting {THROTTLE_WAIT_SECONDS}s before extraction call... ---")
        await asyncio.sleep(THROTTLE_WAIT_SECONDS)
        
        structured_llm = extraction_model.with_structured_output(TranscriptData)
        response_data = await structured_llm.ainvoke(prompt)
        return response_data.model_dump()
    except Exception as e:
        print(f"Error during JSON extraction: {e}")
        return {"events": [], "actions": []}

# --- 4. Database Saving Tools ---

def save_summaries_and_links(summaries: List[Dict[str, Any]]) -> str:
    print(f"--- [Tool] Saving {len(summaries)} summaries and their links to DB... ---")
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cursor:
            for summary_obj in summaries:
                if not summary_obj['summary_embedding']:
                    continue # Skip if embedding failed
                
                cursor.execute(
                    "INSERT INTO summaries (summary_text, embedding) VALUES (%s, %s) RETURNING id",
                    (summary_obj['summary_text'], summary_obj['summary_embedding'])
                )
                summary_id = cursor.fetchone()[0]
                
                for line_id in summary_obj['line_ids']:
                    cursor.execute(
                        "INSERT INTO summary_lines (summary_id, line_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (summary_id, line_id)
                    )
            conn.commit()
            return f"Successfully saved {len(summaries)} summaries."
    except Exception as e:
        conn.rollback()
        print(f"Error in save_summaries_and_links: {e}")
        return f"Error saving summaries: {e}"
    finally:
        conn.close()

def save_and_link_facts(structured_data: dict, parsed_lines: List[ParsedLine]) -> str:
    """
    Saves Events/Actions AND links them back to the lines table
    using robust, normalized string matching.
    """
    print("--- [Tool] Saving and linking facts (events/actions) to DB... ---")
    conn = get_postgres_connection()
    entity_ids_cache = {} 
    try:
        with conn.cursor() as cursor:
            # Process Events
            for event in structured_data.get('events', []):
                loc_id = get_or_create_location(cursor, event.get('location'))
                date_id = get_or_create_id(cursor, "dates", "event_date", event['date'], "date_id")
                
                # 1. INSERT the event
                cursor.execute(
                    "INSERT INTO events (event_name, date_id, location_id) VALUES (%s, %s, %s) RETURNING event_id",
                    (event['event_name'], date_id, loc_id)
                )
                event_id = cursor.fetchone()[0]
                
                # 2. LINK entities to the event
                for name in event['people_involved']:
                    if name not in entity_ids_cache:
                        entity_ids_cache[name] = get_or_create_id(
                            cursor, "entities", "entity_name", name, "entity_id"
                        )
                    entity_id = entity_ids_cache[name]
                    cursor.execute(
                        "INSERT INTO entity_events (entity_id, event_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (entity_id, event_id)
                    )
                
                # 3. LINK the lines to this event
                # --- [!!! THE FIX !!!] ---
                source_sentence = event.get('source_sentence', '')
                normalized_source = normalize_string(source_sentence)
                
                if normalized_source:
                    # Match against the pre-normalized line text
                    matching_line_ids = [
                        line.line_id for line in parsed_lines 
                        if normalized_source in line.normalized_text
                    ]
                    
                    for line_id in matching_line_ids:
                        cursor.execute(
                            "UPDATE lines SET event_id = %s, location_id = %s WHERE line_id = %s",
                            (event_id, loc_id, line_id)
                        )

            # Process Actions
            for action in structured_data.get('actions', []):
                loc_id = get_or_create_location(cursor, action.get('location'))
                date_id = get_or_create_id(cursor, "dates", "event_date", action['date'], "date_id")
                
                # 1. INSERT the action
                cursor.execute(
                    "INSERT INTO actions (action_name, date_id, location_id) VALUES (%s, %s, %s) RETURNING action_id",
                    (action['action_name'], date_id, loc_id)
                )
                action_id = cursor.fetchone()[0]
                
                # 2. LINK entities to the action
                for name in action['people_involved']:
                    if name not in entity_ids_cache:
                        entity_ids_cache[name] = get_or_create_id(
                            cursor, "entities", "entity_name", name, "entity_id"
                        )
                    entity_id = entity_ids_cache[name]
                    cursor.execute(
                        "INSERT INTO entity_actions (entity_id, action_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (entity_id, action_id)
                    )
                
                # 3. LINK the lines to this action

                source_sentence = action.get('source_sentence', '')
                normalized_source = normalize_string(source_sentence)
                
                if normalized_source:
                    matching_line_ids = [
                        line.line_id for line in parsed_lines 
                        if normalized_source in line.normalized_text
                    ]
                    
                    for line_id in matching_line_ids:
                        cursor.execute(
                            "UPDATE lines SET action_id = %s, location_id = %s WHERE line_id = %s",
                            (action_id, loc_id, line_id)
                        )
            
            conn.commit() 
            return "Successfully saved and linked facts."
    except Exception as e:
        conn.rollback() 
        print(f"Error in save_and_link_facts: {e}")
        return f"Error saving/linking facts: {e}"
    finally:
        conn.close()

# --- 5. Build Knowledge Graph from Transcript ---

def extract_graph_from_transcript(transcript_text):
    """
    Uses Google Gemini to extract a structured graph from raw transcript text.
    """
    llm_client = neo4j_gemini_model
    if not llm_client:
        print("LLM client not initialized. Cannot extract data.")
        return None
        
    # The system prompt is now just the first part of the user message.
    # The JSON schema instruction is separate.
    system_prompt = """
    You are an expert NLP and Knowledge Graph analyst. Your task is to read a 
    conversation transcript and extract all specified relationships.
    
    You must extract the following 6 categories:
    1.  **interactions**: Pairs of people who interact or mention knowing each other.
    2.  **preferences**: What a person likes or dislikes.
    3.  **opinions**: A person's opinion on a specific topic or another person.
    4.  **emotions**: A distinct emotion a person expresses (e.g., happy, anxious, frustrated).
    5.  **intents**: The goal or purpose behind a person's statement (e.g., to inform, to persuade).

    **CRITICAL RULE:** If you cannot find a valid value for a required property (like an emotion type, intent type, or opinion target), 
    **you MUST omit the entire object** from the list. 
    Do not create objects with empty strings ("") or null values.

    Return *only* a JSON object matching the requested schema.
    """
    
    # This JSON schema is provided to the model to force its output shape.
    # This is more robust than just putting it in the text prompt.
    json_schema = {
      "type": "object",
      "properties": {
        "interactions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "person1": {"type": "string"},
              "person2": {"type": "string"}
            },
            "required": ["person1", "person2"]
          }
        },
        "preferences": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "person": {"type": "string"},
              "item": {"type": "string"},
              "type": {"type": "string", "enum": ["LIKES", "DISLIKES"]}
            },
            "required": ["person", "item", "type"]
          }
        },
        "opinions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "person": {"type": "string"},
              "target_name": {"type": "string"},
              "target_type": {"type": "string", "enum": ["Topic", "Person"]},
              "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
              "quote": {"type": "string"}
            },
            "required": ["person", "target_name", "target_type", "sentiment"]
          }
        },
        "emotions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "person": {"type": "string"},
              "emotion": {"type": "string"},
              "topic": {"type": "string"}
            },
            "required": ["person", "emotion"]
          }
        },
        "intents": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "person": {"type": "string"},
              "intent": {"type": "string"},
              "target": {"type": "string"},
              "statement": {"type": "string"}
            },
            "required": ["person", "intent"]
          }
        }
      }
    }

    struct_llm_client = llm_client.with_structured_output(
        schema=json_schema,
        method="json_schema"
    )

    print("Sending transcript to Gemini for analysis...")
    try:
        # Create the full prompt
        full_prompt = f"{system_prompt}\n\n**Transcript:**\n{transcript_text}"
        
        # Send the request
        response = struct_llm_client.invoke(full_prompt)
        
        # The model's response.text will be a *string* of valid JSON
        # thanks to the response_mime_type and response_schema.
        return response
        
    except Exception as e:
        print(f"Error during Gemini extraction: {e}")
        # If there's a safety block or other error, print the response
        if 'response' in locals():
            print("Full model response:", response)
        return None
    
# --- 6. Graph Population Nodes ---
def populate_kg(transcript: str):
    print("--- [Tool] Populating knowledge graph from transcript... ---")
    graph_data = None
    NEO4J_URI = get_neo4j_connection()[0]
    NEO4J_USER = get_neo4j_connection()[1]
    NEO4J_PASSWORD = get_neo4j_connection()[2]
    writer = Neo4jWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    if not writer.driver:
        print("Exiting due to Neo4j connection failure.")
        return

    try:
        # 1. Extract data from transcript
        graph_data = extract_graph_from_transcript(transcript)
        
        if not graph_data:
            print("No data extracted from transcript. Exiting.")
            return

        print("\n--- Extracted JSON from LLM ---")
        print(json.dumps(graph_data, indent=2))
        print("---------------------------------")

        # 2. Populate the knowledge graph
        print("\nPopulating Neo4j graph...")
        writer.populate_graph_from_data(graph_data)
        print("Graph population complete.")

    except Exception as e:
        print(f"An error occurred in the main pipeline: {e}")
    finally:
        # 3. Clean up
        print("Closing Neo4j connection.")
        writer.close()