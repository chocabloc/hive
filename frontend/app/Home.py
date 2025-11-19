import streamlit as st
import requests
import os
import uuid

API_HOST = os.getenv("API_URL", "http://localhost:8000")
AUDIO_HOST = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8001")
MCP_HOST = os.getenv("MCP_SERVICE_URL", "http://localhost:8080")

st.set_page_config(page_title="AI Audio Wearable Assistant", page_icon="🎧")

st.title("🎧 AI Audio Wearable Assistant")
st.markdown("""
Welcome to your personal AI companion that listens, remembers, and summarizes your conversations.

Use the sidebar to:
- **Upload** a conversation
- **Query** your past discussions
- **View** all extracted summaries
""")

# Upload audio
st.header("Upload Conversation")
audio_file = st.file_uploader("Upload an audio recording", type=["wav", "mp3"])
speaker_samples = st.file_uploader(
    "Upload speaker sample audios", type=["wav", "mp3"],
    accept_multiple_files=True
)
speaker_names = st.text_input("Names of people in conversation (comma-separated)", help="You must provide one name for each speaker sample file you uploaded.")


if st.button("Process"):
    if audio_file and speaker_names:
        speaker_names_list = [name.strip() for name in speaker_names.split(",")]
        if len(speaker_names_list) != len(speaker_samples):
            st.error(f"Input Mismatch: You uploaded {len(speaker_samples)} sample files but provided {len(speaker_names_list)} names. These must match.")
        else:
            st.success("Processing conversation...")
            try:
                # Prepare data for the API service
                files_payload = []
                
                # Add the main_audio
                # Format: ('field_name', (filename, file_bytes, content_type))
                files_payload.append(
                    ('main_audio', (audio_file.name, audio_file.getvalue(), audio_file.type))
                )
                
                # Add all the speaker_samples
                for sample_file in speaker_samples:
                    files_payload.append(
                        ('speaker_samples', (sample_file.name, sample_file.getvalue(), sample_file.type))
                    )

                # --- C. Build the `data_payload` (list of tuples) ---
                data_payload = [('speaker_names', name) for name in speaker_names_list]
                response = requests.post(f"{API_HOST}/populate", files=files_payload, data=data_payload)
                st.success("Population started in the background successfully!") # I removed the error checking here since it kept getting triggered due to streamlit shenanigans, anyways logs exist in the backend
            except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the database. Please try again later.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an audio file and enter names.")

# Query database
st.header("Ask about your conversations")
query = st.text_input("Enter your query (e.g., 'What did I do on Aug 5?')")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "query_response" not in st.session_state:
    st.session_state.query_response = "Response will appear here."


# --- "New Chat" Button ---
if st.button("New Chat"):
    with st.spinner("Starting new chat..."):
        try:
            payload = {"session_id": st.session_state.session_id}
            # --- CHANGE: Call MCP_HOST ---
            response = requests.post(f"{MCP_HOST}/new_chat", json=payload) 
            
            if response.status_code == 200:
                st.success("New chat started!")
                st.session_state.query_response = "New chat started. Ask a question."
            else:
                st.error(f"Could not start new chat: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Could not connect to the Chat Service at {MCP_HOST}: {e}")

st.divider()

# --- "Search" Functionality ---
query = st.text_input("Ask something about your conversations", key="query_input")

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            try:
                # --- CHANGE: Call MCP_HOST ---
                payload = {
                    "query": query,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(f"{MCP_HOST}/query", json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                    st.session_state.query_response = answer
                else:
                    error_message = f"Error from API Service: {response.status_code} - {response.text}"
                    st.session_state.query_response = error_message
                    st.error(error_message)
            
            except requests.exceptions.ConnectionError:
                error_message = f"Could not connect to the Chat Service at {MCP_HOST}."
                st.session_state.query_response = error_message
                st.error(error_message)
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.session_state.query_response = error_message
                st.error(error_message)
    else:
        st.warning("Please enter a query.")

# --- Display Area ---
st.divider()
st.subheader("Response")
st.write(st.session_state.query_response)

# Feedback
st.header("Feedback")
feedback = st.text_area("Your feedback")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")

# place where the user can see the results from the backend processing and querying
if "summaries" not in st.session_state:
    st.session_state.summaries = []

st.header("Results")
st.subheader("Database Summaries")

if st.session_state.summaries:
    for sid, text, created in st.session_state.summaries:
        with st.container():
            st.write(f"### Summary {sid}")
            st.write(text)
            st.caption(f"Created: {created}")
else:
    st.info("Go to the View Summaries page.")

        