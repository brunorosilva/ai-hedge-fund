#!/usr/bin/env python
"""
AI Product Ideas Evaluator

A tool that evaluates product ideas using AI agents inspired by tech leaders.
"""

import os
import subprocess
import sys
import requests
from dotenv import load_dotenv

def check_ollama_availability():
    """Check if Ollama is running and available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    """Run the AI Product Ideas Evaluator."""
    # Load environment variables
    load_dotenv()
    
    # Check for model provider
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    
    if use_ollama:
        # Check if Ollama is running
        if not check_ollama_availability():
            print("Error: Ollama is not running or not accessible.")
            print("Please start Ollama with 'ollama serve' and make sure it's running on http://localhost:11434")
            sys.exit(1)
        print("Using Ollama for model inference.")
    else:
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is not set.")
            print("Please create a .env file with your OpenAI API key or set USE_OLLAMA=true to use Ollama.")
            sys.exit(1)
        print("Using OpenAI for model inference.")
    
    # Run the Streamlit app
    try:
        subprocess.run(["streamlit", "run", "src/app.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error running the Streamlit app.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with 'pip install streamlit'.")
        sys.exit(1)

if __name__ == "__main__":
    main() 