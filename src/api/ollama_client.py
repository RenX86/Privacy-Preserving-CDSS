"""
Ollama Client Module

PURPOSE:
Interface with the locally running Ollama LLM server.
Handles communication with LLaMA-3, BioMistral, or other local models.

REQUIREMENTS:
- Ollama must be installed and running locally
- Download a model: ollama pull llama3

USAGE:
    from src.api.ollama_client import generate_response, check_ollama_status
    
    response = generate_response("What is DNA?", context="DNA is...")
"""

import requests
from typing import Optional, Generator
import json

from src.config.settings import settings
from src.utils.logger import logger


def check_ollama_status() -> bool:
    """
    Check if Ollama is running and accessible.
    
    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama is running")
            return True
        else:
            logger.warning(f"Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("✗ Cannot connect to Ollama. Is it running?")
        logger.error(f"  Expected at: {settings.OLLAMA_HOST}")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama: {e}")
        return False


def list_models() -> list:
    """
    List available models in Ollama.
    
    Returns:
        List of model names
    """
    try:
        response = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            return models
        return []
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False
) -> str:
    """
    Generate a response from the Ollama LLM.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system instructions
        model: Model to use (default from settings)
        temperature: Creativity (0 = deterministic, 1 = creative)
        max_tokens: Maximum response length
        stream: Whether to stream the response
        
    Returns:
        Generated text response
    """
    model = model or settings.OLLAMA_MODEL
    
    logger.info(f"Generating response with {model}")
    
    # Build the request
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=120  # 2 minute timeout for generation
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama error: {response.status_code} - {response.text}")
            raise Exception(f"Ollama API error: {response.status_code}")
        
        result = response.json()
        generated_text = result.get('response', '')
        
        # Log some stats
        total_duration = result.get('total_duration', 0) / 1e9  # nanoseconds to seconds
        logger.info(f"✓ Generated {len(generated_text)} chars in {total_duration:.2f}s")
        
        return generated_text
        
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        raise
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        raise


def generate_response_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> Generator[str, None, None]:
    """
    Stream a response from Ollama (yields text chunks as they're generated).
    
    Useful for real-time UI updates.
    
    Yields:
        Text chunks as they are generated
    """
    model = model or settings.OLLAMA_MODEL
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        with requests.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        raise


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Ollama Client Test ===\n")
    
    # Check Ollama status
    print("1. Checking Ollama status...")
    if not check_ollama_status():
        print("\nOllama is not running. Start it with:")
        print("  ollama serve")
        print("\nThen install a model:")
        print("  ollama pull llama3")
        exit(1)
    
    # List models
    print("\n2. Available models:")
    models = list_models()
    for model in models:
        print(f"  - {model}")
    
    if not models:
        print("  No models found. Install one with: ollama pull llama3")
        exit(1)
    
    # Test generation
    print(f"\n3. Testing generation with {settings.OLLAMA_MODEL}...")
    test_prompt = "What is DNA? Answer in one sentence."
    
    try:
        response = generate_response(test_prompt, temperature=0.3, max_tokens=100)
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {response}")
        print("\n✓ Ollama client ready!")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        print(f"  Make sure model '{settings.OLLAMA_MODEL}' is installed")
