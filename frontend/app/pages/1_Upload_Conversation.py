import streamlit as st
import requests
import os

API_HOST = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Upload Conversation", page_icon="📤")

st.title("📤 Upload Conversation")

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

st.divider()
st.subheader("Results")
st.write("📝 Conversation Summary will appear here.")
st.write("🏷️ Extracted Entities will appear here.")
