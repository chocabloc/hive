import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import textwrap
import os
import dotenv
from pydub import AudioSegment
from speechbrain.pretrained import EncoderClassifier
import assemblyai as aai

# Load the .env file
dotenv.load_dotenv()

# --- Model Loading ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Audio Service: Loading model on {DEVICE} ---")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# --- Helper Functions ---

def embed_audiosegment(audio_seg, max_len_sec=5):
    """Generates an embedding for a pydub AudioSegment."""
    try:
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(audio_seg.array_type).max

        if audio_seg.channels > 1:
            samples = samples.reshape((-1, audio_seg.channels))
            samples = samples.mean(axis=1)

        waveform = torch.tensor(samples).unsqueeze(0).to(DEVICE)

        if audio_seg.frame_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio_seg.frame_rate, new_freq=16000
            ).to(DEVICE)
            waveform = resampler(waveform)

        max_samples = max_len_sec * 16000
        waveform = waveform[:, :max_samples]
        
        with torch.no_grad():
            speaker_embeddings = classifier.encode_batch(waveform)
            speaker_embeddings = F.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu()
        
        return speaker_embeddings
    except Exception as e:
        print(f"Error embedding audio segment: {e}")
        return None

def weighted_mean_embeddings(embs, weights):
    """Computes a weighted mean of embeddings."""
    if not embs:
        return None
    if len(embs) == 1:
        return embs[0]
    
    W = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    M = torch.stack(embs, dim=0)
    avg = (W * M).sum(dim=0) / (W.sum() + 1e-12)
    return avg / (avg.norm(p=2) + 1e-12)

def build_known_embeddings(known_speakers: dict):
    """Build embeddings for known speakers given a dict {name: audio_path}."""
    known_embeddings = {}
    for name, path in known_speakers.items():
        try:
            seg = AudioSegment.from_file(path)
            emb = embed_audiosegment(seg)
            if emb is not None:
                known_embeddings[name] = emb
        except Exception as e:
            print(f"Could not load or embed speaker file {path}: {e}")
    return known_embeddings

# --- Main Logic ---

def diarize_and_identify(audio_path, known_voice_samples, unknown_threshold=0.55):
    """
    Main logic function:
    1. Reads AAI_API_KEY from environment.
    2. Builds embeddings for known speakers.
    3. Transcribes and diarizes with AssemblyAI.
    4. Matches speakers and returns the final transcript.
    """
    print("Starting diarization and identification...")
    
    # --- 1. Configure AssemblyAI from Environment ---
    api_key = os.getenv("ASSEMBYAI_API_KEY")
    
    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()

    # --- 2. Build Known Embeddings ---
    print(f"Building embeddings for {len(known_voice_samples)} known speakers...")
    known_embeddings = build_known_embeddings(known_voice_samples)
    if not known_embeddings:
        print("Warning: No known speaker embeddings could be built.")

    # --- 3. Transcribe and Diarize ---
    print(f"Uploading {audio_path} to AssemblyAI...")
    upload_url = transcriber.upload_file(audio_path)
    
    config = aai.TranscriptionConfig(speaker_labels=True)
    print("Starting transcription (this may take a moment)...")
    transcript = transcriber.transcribe(upload_url, config)
    
    if transcript.status == aai.TranscriptStatus.error:
        print(f"Error from AssemblyAI: {transcript.error}")
        return f"Error: {transcript.error}"
    if not transcript.utterances:
        print("AssemblyAI returned no utterances.")
        return "No speech was detected in the audio."

    print("Transcription complete. Loading audio for segmentation...")
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        return f"Error loading main audio file: {e}"

    # --- 4. Match Speakers ---
    print("Collecting diarized segments...")
    cluster_segs = {}
    for utt in transcript.utterances:
        try:
            seg = audio[utt.start:utt.end]
            dur = max(0.001, (utt.end - utt.start) / 1000.0)
            cluster_segs.setdefault(utt.speaker, []).append((seg, dur))
        except Exception as e:
            print(f"Warning: Could not segment audio for utterance: {e}")

    print("Computing cluster embeddings...")
    cluster_embeddings = {}
    for label, segs in cluster_segs.items():
        embs, weights = [], []
        for seg, dur in segs:
            emb = embed_audiosegment(seg)
            if emb is not None:
                embs.append(emb)
                weights.append(dur)
        
        cluster_emb = weighted_mean_embeddings(embs, weights)
        if cluster_emb is not None:
            cluster_embeddings[label] = cluster_emb

    print("Matching clusters to known speakers...")
    label_to_name = {}
    for label, emb in cluster_embeddings.items():
        best_name, best_score = None, -1.0
        for name, kemb in known_embeddings.items():
            score = F.cosine_similarity(emb, kemb, dim=0).item()
            if score > best_score:
                best_name, best_score = name, score
        
        if best_score >= unknown_threshold:
            label_to_name[label] = best_name
        else:
            label_to_name[label] = f"Unknown_{label}"

    # --- 5. Build Final Transcript ---
    print("Building final transcript...")
    final_lines = []
    for utt in transcript.utterances:
        who = label_to_name.get(utt.speaker, f"Unknown_{utt.speaker}")
        line = f"[{utt.start/1000:.2f}-{utt.end/1000:.2f}] {who}: {utt.text}"
        final_lines.append(textwrap.fill(line, width=100))

    print("Processing complete.")
    return "\n".join(final_lines)

