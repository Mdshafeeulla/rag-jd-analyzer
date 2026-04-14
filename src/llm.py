# src/llm.py

import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"


def ask_ollama(prompt, model="mistral", temperature=0.3):
    """
    Sends a prompt to the local Ollama LLM and return the response.
    
    Args:
        prompt      : the full RAG prompt (context + JD + instructions)
        model       : ollama model name "mistral"
    
    Returns:
        String response from the LLM
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,           # get full response at once
        "options": {
            "temperature": temperature,
            "num_predict": 1024,   # max tokens in response
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120            # wait up to 2 minutes for response
        )
        response.raise_for_status()
        return response.json()["response"]
    
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure it's running.\n"
            "Open a new terminal and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama took too long to respond. Try a smaller model like 'phi3'.")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def list_available_models():
    """Show which models you've downloaded."""
    try:
        resp = requests.get("http://localhost:11434/api/tags")
        models = resp.json().get("models", [])
        return [m["name"] for m in models]
    except:
        return []


# ── Test it standalone ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Available models:", list_available_models())
    
    print("\nTesting Ollama connection...")
    response = ask_ollama("In one sentence: what is a vector embedding?")
    print(f"Response: {response}")
