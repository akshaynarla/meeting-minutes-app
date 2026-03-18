import os
import time
from src.backend import process_audio, generate_minutes
import requests

os.environ["OLLAMA_STREAM"] = "1"

def get_first_model():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                return models[0]["name"]
    except Exception:
        pass
    return "llama3.1:8b"

def test_transcription():
    audio_path = os.path.join("resources", "Test.m4a")
    print(f"Testing Audio Processing on {audio_path}...")
    
    start_time = time.time()
    result = process_audio(
        audio_path=audio_path,
        model_size="large",  
        device="cpu",       
        compute_type="int8",
        diarize=False
    )
    end_time = time.time()
    print(f"Transcription finished in {end_time - start_time:.2f} seconds.")
    print("Results:", result)
    
    print("\nTesting RAG Summarization on the resulting conversation.md...")
    model_to_use = get_first_model()
    print(f"Using model: {model_to_use}")
    start_time = time.time()
    
    try:
        minutes = generate_minutes(result["conversation"], model=model_to_use)
        end_time = time.time()
        print(f"LLM Minutes generated in {end_time - start_time:.2f} seconds.")
        print("Minutes saved to:", minutes)
    except Exception as e:
        print(f"Failed to generate minutes: {e}")
    
if __name__ == "__main__":
    test_transcription()
