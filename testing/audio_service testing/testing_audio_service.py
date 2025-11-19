import requests
import os


# 1. Define the URL of your running service
url = "http://localhost:8001/process_audio"

# 2. Define the non-file 'data' payload.
# This matches the `speaker_names: list[str] = Form(...)` in your FastAPI app.
# The order here must match the order of the 'speaker_samples' files below.
form_data = [
    ('speaker_names', 'mighty'),
    ('speaker_names', 'guru')
]

# 3. Define the file paths
main_audio_path = 'mighty_guru.mp3'
speaker_1_path = 'mighty.mp3'
speaker_2_path = 'guru.mp3'

# 4. Check if files exist before trying to open them
required_files = [main_audio_path, speaker_1_path, speaker_2_path]
if not all(os.path.exists(f) for f in required_files):
    print(f"--- Error: Missing Test Files ---")
    print(f"Please make sure {', '.join(required_files)} are in this directory.")
    exit()

try:
    # 5. Open all files in 'read-binary' (rb) mode
    # The 'with' block automatically closes the files for you
    with open(main_audio_path, 'rb') as f_main, \
         open(speaker_1_path, 'rb') as f_sp1, \
         open(speaker_2_path, 'rb') as f_sp2:

        # 6. Define the 'files' payload.
        # This matches `main_audio: UploadFile` and `speaker_samples: list[UploadFile]`
        files_list = [
            ('main_audio', (main_audio_path, f_main, 'audio/mpeg')),
            ('speaker_samples', (speaker_1_path, f_sp1, 'audio/mpeg')),
            ('speaker_samples', (speaker_2_path, f_sp2, 'audio/mpeg'))
        ]

        print(f"Sending request to {url}...")
        
        # 7. Send the POST request with both 'data' (names) and 'files' (audio)
        response = requests.post(url, data=form_data, files=files_list)

        # 8. Check the response from the server
        if response.status_code == 200:
            print("\n--- Success! Service Returned: ---")
            # The response.json() will be a dict with 'transcript' and 'details'
            # We'll just print the transcript.
            result_data = response.json()
            print(result_data.get('transcript'))
            
            # Uncomment this line if you want to see the full raw JSON response
            # print(json.dumps(result_data, indent=2))
        else:
            print(f"\n--- Error ---")
            print(f"Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")

except requests.exceptions.ConnectionError:
    print(f"\n--- Connection Error ---")
    print(f"Failed to connect to {url}")
    print("Is your Docker service running? (docker compose up --build)")
except FileNotFoundError as e:
    print(f"\n--- File Not Found ---")
    print(f"Error: {e}")
    print("Please make sure all audio files are in the same directory as this script.")

