import os
import psycopg2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv
# -----------------------
load_dotenv()

# --- Load Environment Variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env file")

# --- Neon DB Credentials ---
NEON_USER = os.getenv("NEON_USER")
NEON_PASSWORD = os.getenv("NEON_PASSWORD")
NEON_DB = os.getenv("NEON_DB")
NEON_HOST = os.getenv("NEON_HOST")

if not all([NEON_USER, NEON_PASSWORD, NEON_DB, NEON_HOST]):
    raise ValueError("One or more NEON_ database variables are not set in .env file")

# --- Neo4j Credentials ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- AI Clients ---

# Client for extraction and summarization
extraction_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-09-2025",
    google_api_key=GEMINI_API_KEY,
    convert_system_message_to_human=True
)

# Client for creating text embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY,
    task_type="retrieval_document"
)

neo4j_gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-09-2025",
        google_api_key=GEMINI_API_KEY
)

# --- Database Client ---
def get_postgres_connection():
    """Establishes a connection to the external Neon PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=NEON_DB,
            user=NEON_USER,
            password=NEON_PASSWORD,
            host=NEON_HOST,
            port="5432",  # Default Postgres port
            sslmode="require" # Required for Neon
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to Neon DB at {NEON_HOST}: {e}")
        raise

# --- Neo4j Client Setup ---
def get_neo4j_connection():
    """Establishes a connection to the Neo4j graph database."""
    def get_driver():
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    def get_logger(name: str) -> logging.Logger:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        logging.basicConfig(level=level, format=fmt)
        return logging.getLogger(name)
    def ping():
        try:
            with get_driver().session() as s:
                s.run("RETURN 1 AS ok").single()
            return True
        except Exception:
            return get_logger(__name__)
    if ping() == True:
        print("Successfully connected to Neo4j database.")
    else:
        raise ConnectionError("Failed to connect to Neo4j database.")
    return NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD