import psycopg2
import streamlit as st
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

NEON_USER = os.getenv("NEON_USER")
NEON_PASSWORD = os.getenv("NEON_PASSWORD")
NEON_DB = os.getenv("NEON_DB")
NEON_HOST = os.getenv("NEON_HOST")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("hello")

def get_postgres_connection_1():
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

conn = get_postgres_connection_1()

print("hello again")

def get_summaries():
    with conn.cursor() as cur:
        cur.execute("SELECT id, summary_text, created_at FROM summaries ORDER BY created_at DESC")
        return cur.fetchall()    

def setup_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY missing in .env")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")
    return model

summary_list = get_summaries()

def unify_summaries(summary_list):
    prompt = """
You are a text summarization model.

You will receive several summaries that describe the same conversation, but each
summary is written in a different format or style.

Your task:
- Combine all the summaries
- Remove redundancy
- Produce ONE clean, cohesive summary
- Use neutral, consistent style
- Preserve all important details
- Ignore formatting or style differences in input
- Final Output should contain high-level overview and key specifics
-Final Output should be in bullet point format and should not contain any text other than the bullet points.
- Keep the final summary under 100 words

Summaries:
""" + "\n\n".join(f"- {s}" for s in summary_list) + """

Final combined summary:
"""
    model = setup_model()
    result = model.generate_content(prompt)
    return result.text.strip()

if "summaries_loaded" not in st.session_state:

    with st.spinner("Loading summaries..."):
        progress = st.progress(0)

        # Step 1: Connect DB
        progress.progress(10)
        conn = get_postgres_connection_1()
        st.session_state.conn = conn
        time.sleep(0.2)

        # Step 2: Fetch rows
        progress.progress(40)
        summary_list = get_summaries()
        time.sleep(0.2)

        # Step 3: Generate combined summary
        progress.progress(70)
        summaries = unify_summaries(summary_list)
        time.sleep(0.2)

        # Step 4: Finalizing
        progress.progress(100)
        st.session_state.summaries = summaries
        st.session_state.summaries_loaded = True

    st.success("Summaries loaded!")

# Use cached summaries
summaries = st.session_state.summaries
if summaries:
    st.write("### Summary")
    st.write(summaries)
else:
    st.info("No summary available.")
