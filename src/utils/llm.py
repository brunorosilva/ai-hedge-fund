"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any, Dict, List
from pydantic import BaseModel
from utils.progress import progress
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from utils.ollama_utils import call_ollama, get_available_models

T = TypeVar('T', bound=BaseModel)

# Load environment variables
load_dotenv()

# Get API key and model configuration
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")

# Initialize LLM if using OpenAI
llm = None
if not use_ollama and api_key:
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key
    )

def call_llm(prompt: Any, output_format: Optional[str] = None) -> Any:
    """
    Call the LLM with a prompt and optionally parse the output.
    
    Args:
        prompt: The prompt to send to the LLM
        output_format: Optional output format specification for JSON parsing
        
    Returns:
        The LLM response, optionally parsed according to the output format
    """
    if use_ollama:
        return call_ollama(prompt, model_name=ollama_model, output_format=output_format)
    elif llm:
        if output_format:
            # Create a parser for the specified output format
            parser = JsonOutputParser()
            
            # Create a prompt template that includes the output format
            prompt_template = ChatPromptTemplate.from_messages([
                prompt.messages[0],
                ("system", f"Format your response as a valid JSON object with the following structure: {output_format}")
            ])
            
            # Create a chain with the prompt, LLM, and parser
            chain = prompt_template | llm | parser
            
            # Run the chain
            return chain.invoke({})
        else:
            # Just run the prompt with the LLM
            return llm.invoke(prompt).content
    else:
        raise ValueError("No LLM available. Please set OPENAI_API_KEY or USE_OLLAMA=true")

def call_llm_with_model(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    # Use Ollama if specified
    if use_ollama or model_provider.lower() == "ollama":
        for attempt in range(max_retries):
            try:
                if agent_name:
                    progress.update_status(agent_name, None, f"Calling Ollama - attempt {attempt + 1}/{max_retries}")
                
                # Get the model schema
                schema = pydantic_model.model_json_schema()
                
                # Call Ollama with the schema
                result = call_ollama(prompt, model_name=ollama_model, output_format=str(schema))
                
                if result and isinstance(result, dict):
                    return pydantic_model(**result)
                
                if attempt == max_retries - 1 and default_factory:
                    return default_factory()
                
            except Exception as e:
                if agent_name:
                    progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
                
                if attempt == max_retries - 1:
                    print(f"Error in Ollama call after {max_retries} attempts: {e}")
                    if default_factory:
                        return default_factory()
                    return create_default_response(pydantic_model)
        
        # Fallback
        return create_default_response(pydantic_model)
    
    # Use OpenAI or other providers
    from llm.models import get_model, get_model_info
    
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            
            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result
                
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)

def extract_json_from_response(content: str) -> Optional[dict]:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None
