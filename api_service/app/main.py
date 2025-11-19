from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from typing import List
import httpx
import os
from .graph_population import population_app
from . import clients

app = FastAPI(
    title="Main API Service",
    description="Handles data population and user queries."
)

# Use a persistent httpx.AsyncClient for performance
http_client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    """Close the httpx client on server shutdown."""
    await http_client.aclose()

async def run_population_graph(transcript: str):
    """
    A wrapper function to run the LangGraph population process.
    This is what will be run in the background.
    """
    print("--- [BG Task] Starting background population graph ---")
    try:
        async for event in population_app.astream({"transcript": transcript}):
            print(f"[BG Task] Graph Event: {event}")
        print("--- [BG Task] Background population graph finished ---")
    except Exception as e:
        print(f"--- [BG Task] ERROR in population graph: {e} ---")
        pass

@app.post("/populate", status_code=status.HTTP_202_ACCEPTED)
async def populate_databases(
    background_tasks: BackgroundTasks, 
    main_audio: UploadFile = File(...),
    speaker_samples: List[UploadFile] = File(...),
    speaker_names: List[str] = Form(...)
):
    
    # --- 1. Prepare data to call audio_service ---
    files_to_send = []
    
    files_to_send.append(
        ('main_audio', (main_audio.filename, await main_audio.read(), main_audio.content_type))
    )
    
    for i, sample in enumerate(speaker_samples):
        files_to_send.append(
            ('speaker_samples', (sample.filename, await sample.read(), sample.content_type))
        )
    
    # --- 2. Call audio_service ---
    
    base_url = os.getenv("AUDIO_SERVICE_URL", "http://audio_service:8001")
    audio_service_url = f"{base_url}/process_audio"
    
    try:
        print(f"--- [Populate] Calling audio_service at {audio_service_url}... ---")
        response = await http_client.post(
            audio_service_url,
            files=files_to_send,
            data={'speaker_names': speaker_names},
            timeout=1200.0
        )
        
        if response.status_code != 200:
            print(f"--- [Populate] Error from audio_service: {response.text} ---")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, 
                detail=f"Audio service failed: {response.text}"
            )
        
    except httpx.RequestError as e:
        print(f"--- [Populate] Cannot connect to audio_service: {e} ---")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, 
            detail=f"Could not connect to audio service: {e}"
        )

    # --- 3. Trigger Background Population Graph ---
    transcript_data = response.json()
    transcript = transcript_data.get('transcript')

    if not transcript:
        raise HTTPException(status_code=400, detail="Audio service returned an empty transcript.")

    background_tasks.add_task(run_population_graph, transcript)
    
    print("--- [Populate] Returning 202 Accepted to user. ---")
    return {
        "message": "Population process started in the background.",
        "transcript_preview": transcript[:200] + "..."
    }