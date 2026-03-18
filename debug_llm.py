import json
import requests
import traceback

def test_raw_ollama():
    print("Testing qwen3.5:4b directly via Ollama API...")
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "Hello! Please summarize this text: 'The dog walked across the street.'"}],
        "options": {"temperature": 0.2, "num_predict": 128},
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {resp.status_code}")
        print(f"Response Body: {resp.text}")
    except Exception as e:
        print(f"Failed: {traceback.format_exc()}")

if __name__ == "__main__":
    test_raw_ollama()
