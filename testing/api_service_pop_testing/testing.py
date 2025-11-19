import requests
import os
import json

# 1. Define the URL of your *api_service*
# We use port 8000, as defined in your docker-compose.yml
url = "http://localhost:8000/populate"

# 2. Define the non-file 'data' payload.
form_data = [
    ('speaker_names', 'mighty'),
    ('speaker_names', 'guru')
]

# 3. Define the file paths
main_audio_path = 'mighty_guru.mp3'
speaker_1_path = 'mighty.mp3'
speaker_2_path = 'guru.mp3'

# --- (Your file check logic is good) ---
required_files = [main_audio_path, speaker_1_path, speaker_2_path]
if not all(os.path.exists(f) for f in required_files):
    print(f"--- Error: Missing Test Files ---")
    print(f"Please make sure {', '.join(required_files)} are in this directory.")
    exit()

try:
    # 4. Open all files in 'read-binary' (rb) mode
    with open(main_audio_path, 'rb') as f_main, \
         open(speaker_1_path, 'rb') as f_sp1, \
         open(speaker_2_path, 'rb') as f_sp2:

        # 5. Define the 'files' payload.
        files_list = [
            ('main_audio', (main_audio_path, f_main, 'audio/mpeg')),
            ('speaker_samples', (speaker_1_path, f_sp1, 'audio/mpeg')),
            ('speaker_samples', (speaker_2_path, f_sp2, 'audio/mpeg'))
        ]

        print(f"Sending request to {url}...")
        
        # 6. Send the POST request
        response = requests.post(url, data=form_data, files=files_list)

        # 7. Check the response from the server
        
        # **IMPORTANT**: We check for 202 (Accepted)
        # This is because your api_service runs the graph in the background
        if response.status_code == 202:
            print("\n--- Success! (202 Accepted) ---")
            print("The API has accepted the request and is processing the data in the background.")
            result_data = response.json()
            print("\nResponse from server:")
            print(json.dumps(result_data, indent=2))
            
            print("\n--- Check your Docker logs ---")
            print("You should now see logs from 'api_service-1' as the LangGraph runs.")
            
        else:
            print(f"\n--- Error ---")
            print(f"Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")

except requests.exceptions.ConnectionError:
    print(f"\n--- Connection Error ---")
    print(f"Failed to connect to {url}")
    print("Is your Docker stack running? (docker compose up)")
except FileNotFoundError as e:
    print(f"\n--- File Not Found ---")
    print(f"Error: {e}")