import os
import json
import requests
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Ollama API endpoint
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

def get_available_models() -> List[str]:
    """Get a list of available models from the Ollama server."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

def call_ollama(prompt: Any, model_name: str = "llama3.1", output_format: Optional[str] = None) -> Any:
    """
    Call the Ollama API with a prompt and optionally parse the output.
    
    Args:
        prompt: The prompt to send to Ollama
        model_name: Name of the model to use (default: llama3.1)
        output_format: Optional output format specification for JSON parsing
        
    Returns:
        The Ollama response, optionally parsed according to the output format
    """
    # Convert prompt to string if it's a ChatPromptTemplate
    if isinstance(prompt, ChatPromptTemplate):
        try:
            # Format the messages with empty kwargs since we don't have any variables to format
            messages = prompt.format_messages()
            prompt_text = ""
            
            for message in messages:
                if isinstance(message, SystemMessage):
                    prompt_text += f"System: {message.content}\n\n"
                elif isinstance(message, HumanMessage):
                    prompt_text += f"Human: {message.content}\n\n"
                else:
                    prompt_text += f"{message.type}: {message.content}\n\n"
        except Exception as e:
            print(f"Error formatting ChatPromptTemplate: {e}")
            prompt_text = str(prompt)
    else:
        prompt_text = str(prompt)
    
    # Add JSON formatting instruction if needed
    if output_format:
        prompt_text += f"\n\nFormat your response as a valid JSON object with the following structure: {output_format}"
    
    # Prepare the request payload
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False
    }
    
    try:
        # Make the API request
        response = requests.post(f"{OLLAMA_API_URL}/generate", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("response", "")
            
            # Parse JSON if needed
            if output_format:
                try:
                    # Try to extract JSON from the response
                    json_content = extract_json_from_response(content)
                    if json_content:
                        return json_content
                except Exception as e:
                    print(f"Error parsing JSON from Ollama response: {e}")
            
            return content
        else:
            print(f"Error calling Ollama API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

def extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from a response string."""
    try:
        # Try to find JSON in the response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        
        # Try to find JSON array
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        
        return None
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
        return None 