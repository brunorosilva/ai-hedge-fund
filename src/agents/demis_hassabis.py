from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
import logging
from typing_extensions import Literal
from utils.llm import call_llm
from utils.progress import progress

# Configure logging
logger = logging.getLogger('ai-product-evaluator')

class DemisHassabisSignal(BaseModel):
    scientific_breakthrough_potential: float  # 0-1 score for scientific innovation
    technical_advancement: float  # 0-1 score for technical advancement
    research_feasibility: float  # 0-1 score for research implementation
    reasoning: str
    key_breakthroughs: list[str]
    research_challenges: list[str]


def demis_hassabis_agent(state: AgentState):
    """Analyzes product ideas using Demis Hassabis's principles and scientific reasoning."""
    data = state["data"]
    product_idea = data["product_idea"]
    technical_context = data.get("technical_context", {})
    research_context = data.get("research_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "scientific_analysis": analyze_scientific_potential(product_idea, research_context),
        "technical_analysis": analyze_technical_advancement(product_idea, technical_context),
        "research_feasibility": analyze_research_feasibility(product_idea),
        "breakthrough_potential": analyze_breakthrough_potential(product_idea, research_context)
    }

    return generate_hassabis_output(product_idea, analysis_data)


def analyze_scientific_potential(product_idea: str, research_context: dict) -> dict:
    """Analyzes scientific potential and innovation opportunities."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze the scientific potential for: {product_idea}
        Consider:
        1. Novel scientific approaches
        2. Research gaps and opportunities
        3. Potential breakthroughs
        4. Scientific impact
        5. Research community interest
        
        Context: {json.dumps(research_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_technical_advancement(product_idea: str, technical_context: dict) -> dict:
    """Analyzes technical advancement and innovation potential."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate technical advancement for: {product_idea}
        Consider:
        1. State-of-the-art technologies
        2. Innovation potential
        3. Technical challenges
        4. Required breakthroughs
        5. Implementation complexity
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_research_feasibility(product_idea: str) -> dict:
    """Analyzes research feasibility and implementation potential."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate research feasibility for: {product_idea}
        Consider:
        1. Required research resources
        2. Timeline estimates
        3. Technical dependencies
        4. Research risks
        5. Validation requirements
        """)
    ])
    
    return call_llm(prompt)


def analyze_breakthrough_potential(product_idea: str, research_context: dict) -> dict:
    """Analyzes potential for scientific breakthroughs and innovations."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze breakthrough potential for: {product_idea}
        Consider:
        1. Novel approaches
        2. Scientific impact
        3. Industry applications
        4. Research community interest
        5. Long-term implications
        
        Context: {json.dumps(research_context)}
        """)
    ])
    
    return call_llm(prompt)


def generate_hassabis_output(product_idea: str, analysis_data: dict) -> DemisHassabisSignal:
    """Generates final output using Demis Hassabis's perspective."""
    # Define the expected output format
    output_format = {
        "scientific_breakthrough_potential": "float between 0 and 1",
        "technical_advancement": "float between 0 and 1",
        "research_feasibility": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_breakthroughs": "list of strings with key breakthroughs",
        "research_challenges": "list of strings with research challenges"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Demis Hassabis, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. Scientific breakthrough potential score (0-1)
        2. Technical advancement score (0-1)
        3. Research feasibility score (0-1)
        4. Detailed reasoning
        5. Key breakthroughs
        6. Research challenges
        
        Format your response as a valid JSON object with the following structure:
        {{
            "scientific_breakthrough_potential": <float between 0 and 1>,
            "technical_advancement": <float between 0 and 1>,
            "research_feasibility": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_breakthroughs": ["<breakthrough 1>", "<breakthrough 2>", ...],
            "research_challenges": ["<challenge 1>", "<challenge 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Demis Hassabis evaluation")
        response = call_llm(prompt, output_format=str(output_format))
        logger.info(f"Raw response from LLM: {response}")
        
        # If response is a string, try to parse it as JSON
        if isinstance(response, str):
            try:
                # Try to extract JSON from the response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    # If no JSON found, create a default response
                    logger.warning("No JSON found in response, creating default")
                    parsed_response = {
                        "scientific_breakthrough_potential": 0.5,
                        "technical_advancement": 0.5,
                        "research_feasibility": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_breakthroughs": ["Default breakthrough"],
                        "research_challenges": ["Default challenge"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                # Create a default response
                parsed_response = {
                    "scientific_breakthrough_potential": 0.5,
                    "technical_advancement": 0.5,
                    "research_feasibility": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_breakthroughs": ["Default breakthrough"],
                    "research_challenges": ["Default challenge"]
                }
        else:
            # Response is already a dict
            parsed_response = response
        
        # Ensure all required fields are present
        if "scientific_breakthrough_potential" not in parsed_response:
            parsed_response["scientific_breakthrough_potential"] = 0.5
        if "technical_advancement" not in parsed_response:
            parsed_response["technical_advancement"] = 0.5
        if "research_feasibility" not in parsed_response:
            parsed_response["research_feasibility"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_breakthroughs" not in parsed_response:
            parsed_response["key_breakthroughs"] = ["Default breakthrough"]
        if "research_challenges" not in parsed_response:
            parsed_response["research_challenges"] = ["Default challenge"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return DemisHassabisSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Demis Hassabis output: {e}", exc_info=True)
        # Return a default response
        return DemisHassabisSignal(
            scientific_breakthrough_potential=0.5,
            technical_advancement=0.5,
            research_feasibility=0.5,
            reasoning="Error occurred during evaluation.",
            key_breakthroughs=["Default breakthrough"],
            research_challenges=["Default challenge"]
        ) 