Most of the codes are poorly commented/structured. The final working version is `obtain_json2.ipynb` - it inputs a transcript and outputs a json in the required  format.  
We use LLMs for summarization and extracting factual information. In particular a Mistral 7B hosted on HuggingFace is used.  
What the other files contain:
- `embedding.ipynb`: experimenting embedding various stuff. We use embeddings to flag important dialogues and also later during test time - to pull out relevant summaries.
- `offline_summary.ipynb`: trying to run Mistral 7B locally. Crashes frequently due to insufficient VRAM.
- `summary.ipynb`: experimenting with getting an online hosted model to generate summaries of transcripts.
- `obtain_json.ipynb`: precursor to `obtain_json.ipynb`. Very messy.
