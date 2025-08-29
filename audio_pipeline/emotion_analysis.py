#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Dict, Any

import torch
from transformers import pipeline

MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

def load_pipeline(device_preference: str):

    if device_preference == "cuda" and torch.cuda.is_available():
        device = 0
    else:
        device = -1  # CPU
    clf = pipeline(
        task="audio-classification",
        model=MODEL_ID,
        sampling_rate=16_000,
        device=device,
        top_k=None,  
    )
    return clf, device

def run_on_files(clf, paths: List[str]) -> Dict[str, Any]:
    results = {}
    for p in paths:
        if not os.path.exists(p):
            results[p] = {"error": "file not found"}
            continue
        try:
            out = clf(p)
            out = sorted(out, key=lambda x: x["score"], reverse=True)
            results[p] = {
                "top_label": out[0]["label"],
                "top_score": float(out[0]["score"]),
                "all_scores": [{k: (float(v) if k == "score" else v) for k, v in d.items()} for d in out],
            }
        except Exception as e:
            results[p] = {"error": str(e)}
    return results

def pretty_print(results: Dict[str, Any], as_json: bool):
    if as_json:
        print(json.dumps(results, indent=2))
        return
    # human-readable
    for path, res in results.items():
        print(f"\n=== {path} ===")
        if "error" in res:
            print(f"ERROR: {res['error']}")
            continue
        print(f"Predicted: {res['top_label']}  (score={res['top_score']:.4f})")
        for item in res["all_scores"]:
            print(f" - {item['label']:10s} : {item['score']:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Speech Emotion Recognition (wav2vec2 XLSR)")
    ap.add_argument("audio", nargs="+", help="Path(s) to audio files (wav/mp3/flac/ogg etc.)")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                    help="Force device. Default: auto (CUDA>MPS>CPU).")
    ap.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = ap.parse_args()

    # Resolve device preference
    pref = args.device
    if pref == "auto":
        pref = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    clf, device = load_pipeline(pref)
    if device == -1:
        sys.stderr.write("Running on CPU.\n")
    elif device == 0:
        sys.stderr.write("Running on CUDA GPU.\n")

    results = run_on_files(clf, args.audio)
    pretty_print(results, args.json)

if __name__ == "__main__":
    main()
