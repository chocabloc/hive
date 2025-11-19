Contains all the evaluation results along with the dataset and the code used to evaluate the models. Run evaluator() along with the model name and path on the golden dictionary. load_model() loads models from google/llama.cpp/openrouter, just add the required model to the model dictionary with the same format.

chunk_transcript() only works with local models as gemini does not expose its tokenizer. Gemma4b was loaded locally to chunk transcripts for gemini due to the similarity in their tokenizers.
