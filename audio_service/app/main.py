import os
import shutil
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from .processing import diarize_and_identify 
# -----------------------

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Model is loaded in processing.py at the global scope.
    print("--- Audio Service: Model loaded. Ready for requests. ---")

@app.post("/process_audio")
async def process_audio_endpoint(
    main_audio: UploadFile = File(...),
    speaker_samples: List[UploadFile] = File(...),
    speaker_names: List[str] = Form(...)
):
    
    if len(speaker_samples) != len(speaker_names):
        raise HTTPException(
            status_code=400,
            detail="The number of speaker samples does not match the number of speaker names."
        )

    temp_dir = tempfile.mkdtemp()
    known_speaker_paths = {}

    try:
        # --- 1. Save the main audio file ---
        main_audio_ext = os.path.splitext(main_audio.filename)[1]
        main_audio_path = os.path.join(temp_dir, f"main{main_audio_ext}")
        
        with open(main_audio_path, "wb") as f:
            shutil.copyfileobj(main_audio.file, f)
        
        # --- 2. Save all speaker sample files ---
        for i, sample_file in enumerate(speaker_samples):
            speaker_name = speaker_names[i]
            sample_ext = os.path.splitext(sample_file.filename)[1]
            sample_path = os.path.join(temp_dir, f"{speaker_name}_{i}{sample_ext}")
            
            with open(sample_path, "wb") as f:
                shutil.copyfileobj(sample_file.file, f)
            
            known_speaker_paths[speaker_name] = sample_path

        # --- 3. Run the processing ---
        result_transcript = diarize_and_identify(
            audio_path=main_audio_path,
            known_voice_samples=known_speaker_paths
        )
        
        return {"transcript": result_transcript}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # --- 4. Clean up the temporary directory ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)