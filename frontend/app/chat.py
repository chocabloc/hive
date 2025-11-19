import asyncio
import json
import os
import re
import logging
import argparse  # Added argparse
from typing import Optional, Dict, Any, List

from fastmcp.client import Client
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.generativeai.generative_models import GenerativeModel, ChatSession
from dotenv import load_dotenv

# --- Constants ---

load_dotenv()

# Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"
MCP_URL = "http://127.0.0.1:8080/sse"
MAX_TOOL_ITERATIONS = 10

# --- System Prompt Definition ---

# Combine the parts into the final prompt
CODE_TICKS = "```"

SYSTEM_PROMPT_PART_1 = """
You are a personal AI assistant designed to help users with past conversations they have been a part of.
You have access to some conversation data which has been extracted from past meeting transcripts.

There is a PostgreSQL database storing entities (people), events, actions, locations, dates, conversation lines, semantic summaries, etc.
The exact schema used to create the database is as follows:
"""

SYSTEM_PROMPT_SQL_SCHEMA = f"""
{CODE_TICKS}
CREATE TABLE IF NOT EXISTS entities(
    entity_id SERIAL PRIMARY KEY,
    entity_name TEXT NOT NULL 
);
CREATE TABLE IF NOT EXISTS locations( 
    location_id SERIAL PRIMARY KEY, 
    country TEXT NOT NULL, 
    city TEXT NOT NULL 
); 
    
CREATE TABLE IF NOT EXISTS dates ( 
    date_id SERIAL PRIMARY KEY, 
    event_date TEXT NOT NULL 
); 
    
CREATE TABLE IF NOT EXISTS actions( 
    action_id SERIAL PRIMARY KEY, 
    action_name TEXT NOT NULL, 
    date_id INT REFERENCES dates(date_id),
    location_id INT REFERENCES locations(location_id) 
); 
    
CREATE TABLE IF NOT EXISTS events( 
    event_id SERIAL PRIMARY KEY, 
    event_name TEXT NOT NULL, 
    date_id INT REFERENCES dates(date_id),
    location_id INT REFERENCES locations(location_id) 
); 

CREATE TABLE IF NOT EXISTS lines(
    line_id SERIAL PRIMARY KEY, 
    line_text TEXT NOT NULL, 
    spokentime TIMESTAMPTZ NOT NULL, 
    entity_id INT REFERENCES entities(entity_id), 
    action_id INT REFERENCES actions(action_id), 
    event_id INT REFERENCES events(event_id), 
    location_id INT REFERENCES locations(location_id) 
); 
-- Many to many links mentioned in the schema (Linking Tables) 
-- persons to events 

CREATE TABLE IF NOT EXISTS entity_events (
    entity_id INT NOT NULL REFERENCES entities(entity_id),
    event_id INT NOT NULL REFERENCES events(event_id),
    PRIMARY KEY (entity_id, event_id)
);
-- persons to actions 

CREATE TABLE IF NOT EXISTS entity_actions (
    entity_id INT NOT NULL REFERENCES entities(entity_id),
    action_id INT NOT NULL REFERENCES actions(action_id),
    PRIMARY KEY (entity_id, action_id)
);


CREATE EXTENSION IF NOT EXISTS vector; 
-- Summaries table 
CREATE TABLE IF NOT EXISTS summaries ( 
    id SERIAL PRIMARY KEY, 
    summary_text TEXT NOT NULL, 
    
    embedding VECTOR(768) NOT NULL, -- 768-dim embedding vector
    
    created_at TIMESTAMPTZ DEFAULT NOW() 
    
); 
-- Linking table to connect summaries to lines 
CREATE TABLE IF NOT EXISTS summary_lines ( 
    summary_id INT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE, 
    line_id INT NOT NULL REFERENCES lines(line_id) ON DELETE CASCADE, 
    PRIMARY KEY (summary_id, line_id) 
    
);
{CODE_TICKS}
"""

SYSTEM_PROMPT_PART_2 = """
There is a knowledge graph, with schema as follows:
- Person nodes capture speakers; edges: `(:Person)-[:KNOWS]->(:Person)` and `(:Person)-[:INTERACTS_WITH {count}] ->(:Person)`.
- Preferences: `(:Person)-[:LIKES|:DISLIKES]->(:Item)`.
- Opinions: `(:Person)-[:HAS_OPINION {sentiment, quote}]->(:Topic or :Person)`.
- Emotions: `(:Person)-[:FEELS]->(:Emotion {type, topic})`.
- Intents: `(:Person)-[:HAD_INTENT {statement}]->(:Goal {type, target})`.

You have the following tools available:
1. execute_sql(query: str)
   - Executes a raw SQL SELECT query.
2. semantic_search(query: str, limit: int = 20, min_similarity: float = 0.3)
    - Computes semantic similarity using pgvector against summaries.embedding.
    - Returns:
       summary_id, summary_text, similarity, and all linked lines (via summary_lines),
       including entity_name, action_name, event_name, location.
3. query_knowledge_graph(cypher: str, limit: int = 100)
    - Runs read-only Cypher against the Neo4j graph.
    - Returns node and relationship objects (id, labels/type, properties, endpoints).
    - MASSIVE WARNING: the knowledge graph may have incomplete/missing information.
    - do not use this as your only source of information.
4. search_database(query: str, limit: int = 50)
   - Performs an exact string search (case insensitive) across lines.line_text, entities.entity_name, events.event_name, actions.action_name, locations.country and locations.city
   - Returns JSON objects with fields:
       result_type, id, content,
       spokentime, entity_name, action_name, event_name, location

Your task:
You must answer user questions. using whatever tool calls you need and as much time as you want.
Call any tools as many times as needed to gather information before answering the user's question.

When you need to call a tool, respond with ONLY a JSON object in this EXACT format (no markdown, no code fences or triple ticks, no explanations):
{
  "action": "tool_call",
  "tool": "<tool_name>",
  "args": {
    ... arguments ...
  }
}

When you have enough information to answer the user, respond with ONLY a JSON object in this EXACT format (make sure it is valid JSON, escape characters in the answer as needed):
{
    "action": "answer",
    "answer": "<your final english answer to the user>"
}

If there is an error with your tool call or answer format, the system will relay it to you, and then you MUST try again.
The database, the knowledge graph, and the existence of tools are simply implementation details, do not mention them in the final answer.

IMPORTANT:
Remember not to expose implementation details to the user.
Be resourceful and intelligent. If you can't find information using one tool, try another approach without asking the user.
DO NOT give up unless you're sure that you cannot find the information using _any_ tool calls.
"""

# Combine the parts into the final prompt
SYSTEM_PROMPT = (
    SYSTEM_PROMPT_PART_1
    + SYSTEM_PROMPT_SQL_SCHEMA
    + SYSTEM_PROMPT_PART_2
)

# Initial history to set up the context for the model
INITIAL_HISTORY = [
    {"role": "user", "parts": [SYSTEM_PROMPT]},
    {
        "role": "model",
        "parts": [
            "Understood. I will help you query the database using the "
            "available tools. I'll respond with either JSON containing a "
            "tool call, or a final answer."
        ],
    },
]

# --- Helper Functions ---

def setup_model() -> GenerativeModel:
    """Configure and return the Gemini model."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Set generation config to ensure JSON output if possible,
    # though the prompt is the main driver here.
    config = GenerationConfig(response_mime_type="application/json")
    
    model = genai.GenerativeModel(
        GEMINI_MODEL_NAME
        # generation_config=config  # Note: Enable this if your model/version supports it well.
                                  # For now, we rely on the prompt and regex parsing.
    )
    return model


async def call_mcp_tool(tool_name: str, args: dict, debug_mode: bool = False) -> Any:
    """Call an MCP tool and return the result."""
    if debug_mode:
        print(f"[Calling {tool_name} with args: {args}]")
    else:
        print(f"[Calling {tool_name}]")
    try:
        async with Client(MCP_URL) as mcp_client:
            result = await mcp_client.call_tool(tool_name, args)
            
            if hasattr(result, 'content') and result.content:
                result_text = result.content[0].text
                try:
                    # Attempt to parse the result as JSON
                    parsed_json = json.loads(result_text)
                    if debug_mode:
                        print(f"✓ Tool Result (JSON):\n{json.dumps(parsed_json, indent=2)}")
                    return parsed_json
                except json.JSONDecodeError:
                    # Fallback to text if not valid JSON
                    if debug_mode:
                        print(f"✓ Tool Result (Text):\n{result_text}")
                    return result_text
            
            print("✓ Tool call returned no content.")
            return str(result)
            
    except Exception as e:
        print(f"❌ Error calling tool {tool_name}: {e}")
        return {"error": f"Failed to call tool {tool_name}: {str(e)}"}


def parse_model_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse the model's response text to find and load a JSON object.
    This handles markdown fences (```json ... ```) and other surrounding text.
    """
    # Regex to find a JSON object or array, ignoring potential markdown fences
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response_text)
    
    if not json_match:
        # logging.warning(f"No JSON object found in model response: {response_text}")
        return None
        
    json_str = json_match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from response: {e}")
        logging.error(f"Invalid JSON string was: {json_str}")
        return None


def format_final_answer(payload: Any) -> str:
    """Convert a final JSON payload into user-friendly text."""
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("answer", "text", "message", "response"):
            value = payload.get(key)
            if isinstance(value, str):
                return value

        # If it looks like a single key/value pair, format it nicely
        if len(payload) == 1:
            key, value = next(iter(payload.items()))
            if isinstance(value, (str, int, float)):
                return f"{key}: {value}"

    # Fallback to pretty JSON
    try:
        return json.dumps(payload, indent=2)
    except TypeError:
        return str(payload)

# --- Main Application Logic ---

async def main():
    """Main chat loop."""

    # --- Add command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Database Chat Interface")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging, including tool outputs"
    )
    args = parser.parse_args()
    debug_mode = args.debug

    # --- Configure logging based on debug flag ---
    # Silence noisy httpx logs only if NOT in debug mode
    if not debug_mode:
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
    if debug_mode:
        print("--- DEBUG MODE ENABLED ---")
    
    try:
        model = setup_model()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return

    print("=" * 60)
    print("Database Chat Interface")
    print("=" * 60)
    print("Ask questions about your database. Type 'exit' to quit.\n")
    
    # Create the stateful chat session ONCE, with the initial context.
    chat_session = model.start_chat(history=INITIAL_HISTORY)

    while True:
        try:
            # 1. Get user input
            user_query = input("\nYou: ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_query:
                continue

            # 2. Send user message to the stateful chat session
            # The session already has the system prompt and all previous turns.
            current_response = chat_session.send_message(user_query)

            # 3. Handle the model's response (which could be an answer or tool call)
            # This loop will run until an "answer" is found or max iterations are hit.
            invalid_json_attempts = 0
            for _ in range(MAX_TOOL_ITERATIONS):
                response_text = current_response.text
                response_json = parse_model_response(response_text)

                if not response_json:
                    invalid_json_attempts += 1
                    logging.warning("Model response was not valid JSON. Attempt %s", invalid_json_attempts)

                    if invalid_json_attempts >= 3:
                        print("\nAssistant: I ran into a formatting issue while processing that request. Please ask again.")
                        break

                    # Nudge the model with a reminder and continue without exposing the error to the user
                    current_response = chat_session.send_message(
                        "Reminder: respond with valid JSON per the system instructions."
                    )
                    continue
                else:
                    invalid_json_attempts = 0
                    # If the model returned a JSON object, it may be a tool call
                    # or a final JSON answer. We recognize tool calls by the
                    # explicit action field: {"action":"tool_call", ...}
                    action = response_json.get("action")

                    if action == "tool_call":
                        tool_name = response_json.get("tool")
                        tool_args = response_json.get("args", {})

                        if not tool_name:
                            # Relay error back to the model and inform the user
                            err = {"error": "tool_call requested but no 'tool' provided"}
                            print(f"\nAssistant error: {err['error']}")
                            current_response = chat_session.send_message(f"Tool result: {json.dumps(err)}")
                            break

                        # Call the tool, passing debug_mode
                        tool_result = await call_mcp_tool(tool_name, tool_args, debug_mode=debug_mode)

                        # If the tool returned an error object, print it for the user
                        if isinstance(tool_result, dict) and tool_result.get("error"):
                            print(f"\nTool error: {tool_result.get('error')}")

                        # Send the tool result back to the *same session* so the model can continue
                        tool_result_str = f"Tool result: {json.dumps(tool_result, default=str)}"
                        current_response = chat_session.send_message(tool_result_str)

                        # Continue the tool loop to process the model's new response
                        continue

                    else:
                        # Treat any other JSON response as a final answer. Relay
                        # error objects explicitly, otherwise pretty-print JSON.
                        if isinstance(response_json, dict) and response_json.get("error"):
                            print(f"\nAssistant (error): {response_json.get('error')}")
                        else:
                            final_text = format_final_answer(response_json)
                            print(f"\nAssistant: {final_text}")
                        break
            else:
                # This 'else' triggers if the 'for' loop completes without 'break'
                print("\nMaximum tool iterations reached. Please try rephrasing your question.")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logging.exception("Chat loop error")
            # Consider whether to break or continue
            # For robustness, we'll continue
            print("Restarting conversation loop. Please try your query again.")
            # Re-initialize the chat session to start fresh
            chat_session = model.start_chat(history=INITIAL_HISTORY)


if __name__ == "__main__":
    # Set the default logging level
    logging.basicConfig(level=logging.INFO)
    
    # --- CHANGE: Silence noisy httpx logs only if NOT in debug mode ---
    # This logic is now moved inside main() so it can access the --debug flag
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")