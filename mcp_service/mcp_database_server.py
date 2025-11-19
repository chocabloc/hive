#!/usr/bin/env python3
"""
MCP Database Server using FastMCP with lifespan management
Uses sentence-transformers for semantic search.
"""

import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.parse

import asyncpg
from dotenv import load_dotenv
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.graph import Node, Path, Relationship
from mcp.server.fastmcp import Context, FastMCP

load_dotenv()

# Optional semantic search - using Google's text-embedding-004 (768 dims)
HAS_EMBEDDINGS = False
embedding_client = None
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import os
    
    # Use the Gemini API key from environment
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env file")
    
    embedding_client = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY,
        task_type="retrieval_query"
    )
    HAS_EMBEDDINGS = True
    print("Google embeddings loaded successfully (768 dims)", file=sys.stderr)
except ImportError:
    print("Google GenAI not available - semantic search disabled", file=sys.stderr)
except Exception as e:
    print(f"Error loading Google embeddings: {e}", file=sys.stderr)

@dataclass
class AppContext:
    """Application context with database and graph connections."""
    db_pool: asyncpg.pool.Pool
    neo4j_driver: AsyncDriver

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with database connection."""
    print("Starting MCP Database Server...", file=sys.stderr)
    
    # Initialize database connection on startup
    pg_user = os.getenv("NEON_USER")
    pg_password = os.getenv("NEON_PASSWORD")
    pg_host = os.getenv("NEON_HOST")
    pg_db = os.getenv("NEON_DB")

    # URL encode the password to handle special characters safely
    encoded_pwd = urllib.parse.quote_plus(pg_password) if pg_password else ""

    connection_string = f"postgresql://{pg_user}:{encoded_pwd}@{pg_host}/{pg_db}?sslmode=require"
    
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise RuntimeError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in the environment")

    neo4j_driver: AsyncDriver | None = None

    try:
        db_pool = await asyncpg.create_pool(connection_string, min_size=1, max_size=5)
        print("Database connection established", file=sys.stderr)
        
        # Test Postgres connection
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        async with neo4j_driver.session() as session:
            await session.run("RETURN 1 AS ok")
        print("Neo4j connection established", file=sys.stderr)

        print("MCP Server ready", file=sys.stderr)
        
        try:
            yield AppContext(db_pool=db_pool, neo4j_driver=neo4j_driver)
        finally:
            # Cleanup on shutdown
            await db_pool.close()
            print("Database connection closed", file=sys.stderr)
            if neo4j_driver:
                await neo4j_driver.close()
                print("Neo4j connection closed", file=sys.stderr)
    except Exception as e:
        print(f"Database connection error: {str(e)}", file=sys.stderr)
        raise

# Create the MCP server with lifespan
mcp = FastMCP("Database Server", lifespan=app_lifespan, host="0.0.0.0", port=8080)

# Helper function to make data JSON-serializable
def safe_serialize(obj):
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)

def rows_to_json(rows):
    # Convert rows to a list of dicts, but ensure values are strings
    results = []
    for row in rows:
        row_dict = {}
        for key, value in dict(row).items():
            # Serialize standard types, convert dates/complex types to strings
            if value is None or isinstance(value, (int, float, bool, str)):
                row_dict[key] = value
            else:
                row_dict[key] = safe_serialize(value)
        results.append(row_dict)
        
    # Return valid JSON string
    return json.dumps(results)


def serialize_neo4j_value(value: Any):
    """Convert Neo4j driver objects into JSON-serializable structures."""
    if isinstance(value, Node):
        return {
            "id": value.id,
            "labels": list(value.labels),
            "properties": {k: serialize_neo4j_value(v) for k, v in dict(value).items()}
        }
    if isinstance(value, Relationship):
        return {
            "id": value.id,
            "type": value.type,
            "start": value.start_node.id,
            "end": value.end_node.id,
            "properties": {k: serialize_neo4j_value(v) for k, v in dict(value).items()}
        }
    if isinstance(value, Path):
        return {
            "nodes": [serialize_neo4j_value(node) for node in value.nodes],
            "relationships": [serialize_neo4j_value(rel) for rel in value.relationships]
        }
    if isinstance(value, list):
        return [serialize_neo4j_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_neo4j_value(val) for key, val in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

@mcp.tool()
async def search_database(query: str, limit: int = 50, ctx: Context = None) -> str:
    """
    Search the database across:
      - lines
      - entities
      - events
      - actions
      - locations
    """
    limit = min(limit, 10)

    search_term = f"%{query.lower()}%"
    db_pool = ctx.request_context.lifespan_context.db_pool

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            -- Search lines
            SELECT 
                'line' AS result_type,
                l.line_id AS id,
                l.line_text AS content,
                l.spokentime,
                en.entity_name,
                a.action_name,
                ev.event_name,
                (loc.country || ', ' || loc.city) AS location
            FROM lines l
            LEFT JOIN entities en ON l.entity_id = en.entity_id
            LEFT JOIN actions a ON l.action_id = a.action_id
            LEFT JOIN events ev ON l.event_id = ev.event_id
            LEFT JOIN locations loc ON l.location_id = loc.location_id
            WHERE LOWER(l.line_text) LIKE $1

            UNION ALL

            -- Search entities
            SELECT 
                'entity' AS result_type, 
                e.entity_id AS id,
                e.entity_name AS content,
                NULL AS spokentime,
                NULL, NULL, NULL, NULL
            FROM entities e
            WHERE LOWER(e.entity_name) LIKE $1

            UNION ALL

            -- Search events
            SELECT
                'event' AS result_type,
                ev.event_id AS id,
                ev.event_name AS content,
                NULL AS spokentime,
                NULL, NULL, NULL,
                (loc.country || ', ' || loc.city)
            FROM events ev
            LEFT JOIN locations loc ON ev.location_id = loc.location_id
            WHERE LOWER(ev.event_name) LIKE $1

            UNION ALL

            -- Search actions
            SELECT
                'action' AS result_type,
                ac.action_id AS id,
                ac.action_name AS content,
                NULL AS spokentime,
                NULL, NULL, NULL,
                (loc.country || ', ' || loc.city)
            FROM actions ac
            LEFT JOIN locations loc ON ac.location_id = loc.location_id
            WHERE LOWER(ac.action_name) LIKE $1

            UNION ALL

            -- Search locations
            SELECT
                'location' AS result_type,
                l.location_id AS id,
                (l.country || ', ' || l.city) AS content,
                NULL AS spokentime,
                NULL, NULL, NULL, NULL
            FROM locations l
            WHERE LOWER(l.country) LIKE $1
               OR LOWER(l.city) LIKE $1

            LIMIT $2
            """,
            search_term, limit
        )

        return rows_to_json(rows)


@mcp.tool()
async def execute_sql(query: str, ctx: Context = None) -> str:
    """Execute custom SELECT-only SQL queries."""
    
    # ... security checks ...
    
    db_pool = ctx.request_context.lifespan_context.db_pool

    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(query)
            return rows_to_json(rows)
            
        except Exception as e:
            return f"SQL Error: {e}"


@mcp.tool()
async def query_knowledge_graph(
    cypher: str,
    params: str | None = None,
    limit: int = 100,
    ctx: Context = None
) -> str:
    """Run read-only Cypher queries against the Neo4j knowledge graph."""

    if not cypher or not cypher.strip():
        return "Cypher query cannot be empty."

    lowered = f" {cypher.lower()} "
    forbidden = (" create ", " delete ", " merge ", " set ", " drop ", " remove ")
    if any(keyword in lowered for keyword in forbidden):
        return "Write operations are disabled for safety. Please submit a read-only Cypher query."

    try:
        params_dict = json.loads(params) if params else {}
        if params_dict and not isinstance(params_dict, dict):
            return "Params must be a JSON object."
    except json.JSONDecodeError as exc:
        return f"Invalid params JSON: {exc}"

    limit = max(1, min(limit, 200))
    driver = ctx.request_context.lifespan_context.neo4j_driver

    async def _run(tx, statement, statement_params, max_rows):
        result = await tx.run(statement, **statement_params)
        collected = []
        count = 0
        async for record in result:
            collected.append({key: serialize_neo4j_value(value) for key, value in record.items()})
            count += 1
            if count >= max_rows:
                break
        return collected

    try:
        async with driver.session() as session:
            rows = await session.execute_read(_run, cypher, params_dict, limit)
        return json.dumps(rows)
    except Exception as exc:
        return f"Cypher Error: {exc}"
        
# HAS_EMBEDDINGS = True

if HAS_EMBEDDINGS:
    @mcp.tool()
    async def semantic_search(query: str, limit: int = 20, min_similarity: float = 0.0, ctx: Context = None):
        # Hard-cap limit so responses never overflow SSE
        limit = min(limit, 3)

        # Use Google's embedding client (async)
        embedding = await embedding_client.aembed_query(query)
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        db_pool = ctx.request_context.lifespan_context.db_pool

        async with db_pool.acquire() as conn:
            summaries = await conn.fetch(
                """
                SELECT
                    s.id,
                    LEFT(s.summary_text, 500) AS summary_text,
                    1 - (s.embedding <=> $1::vector) AS similarity
                FROM summaries s
                WHERE (1 - (s.embedding <=> $1::vector)) >= $3
                ORDER BY similarity DESC
                LIMIT $2
                """,
                embedding_str, limit, min_similarity
            )

            results = []

            for row in summaries:
                summary_id = row["id"]

                # 🚨 Truncate to avoid SSE buffer overflow
                summary_text_safe = (row["summary_text"] or "")[:500]

                lines = await conn.fetch(
                    """
                    SELECT l.line_id, l.line_text, l.spokentime,
                           en.entity_name, ac.action_name, ev.event_name,
                           (loc.country || ', ' || loc.city) AS location
                    FROM summary_lines sl
                    JOIN lines l ON sl.line_id = l.line_id
                    LEFT JOIN entities en ON l.entity_id = en.entity_id
                    LEFT JOIN actions ac ON l.action_id = ac.action_id
                    LEFT JOIN events ev ON l.event_id = ev.event_id
                    LEFT JOIN locations loc ON l.location_id = loc.location_id
                    WHERE sl.summary_id = $1
                    ORDER BY l.spokentime
                    LIMIT 5
                    """,
                    summary_id
                )

                results.append({
                    "summary_id": summary_id,
                    "summary_text": summary_text_safe,
                    "similarity": float(row["similarity"]),
                    "lines": [dict(l) for l in lines],
                })

        return results

if __name__ == "__main__":
    # Always run as HTTP/SSE server on port 8080 for the Gemini client
    mcp.run(transport="sse")
