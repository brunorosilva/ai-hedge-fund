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

class SamAltmanSignal(BaseModel):
    opportunity_score: float  # 0-1 score for opportunity
    market_potential: float  # 0-1 score for market size
    technical_feasibility: float  # 0-1 score for technical implementation
    reasoning: str
    key_insights: list[str]
    potential_risks: list[str]


def sam_altman_agent(state: AgentState):
    """Analyzes product ideas using Sam Altman's principles and LLM reasoning."""
    data = state["data"]
    product_idea = data["product_idea"]
    market_context = data.get("market_context", {})
    technical_context = data.get("technical_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "market_analysis": analyze_market_opportunity(product_idea, market_context),
        "technical_analysis": analyze_technical_feasibility(product_idea, technical_context),
        "scaling_potential": analyze_scaling_potential(product_idea),
        "competitive_analysis": analyze_competitive_landscape(product_idea, market_context)
    }
    logger.info(f"Analysis data: {analysis_data}")
    return generate_altman_output(product_idea, analysis_data)


def analyze_market_opportunity(product_idea: str, market_context: dict) -> dict:
    """Analyzes market opportunity using Altman's principles."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze the market opportunity for: {product_idea}
        Consider:
        1. Total addressable market
        2. Market growth rate
        3. Customer pain points
        4. Willingness to pay
        5. Market timing
        
        Context: {json.dumps(market_context)}
        """)
    ])
    logger.info(f"Prompt: {prompt}")
    return call_llm(prompt)


def analyze_technical_feasibility(product_idea: str, technical_context: dict) -> dict:
    """Analyzes technical feasibility of the product idea."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate technical feasibility for: {product_idea}
        Consider:
        1. Required technologies
        2. Development complexity
        3. Resource requirements
        4. Technical risks
        5. Time to market
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_scaling_potential(product_idea: str) -> dict:
    """Analyzes scaling potential using Altman's growth principles."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate scaling potential for: {product_idea}
        Consider:
        1. Network effects
        2. Viral potential
        3. Customer acquisition costs
        4. Revenue model scalability
        5. Operational scalability
        """)
    ])
    
    return call_llm(prompt)


def analyze_competitive_landscape(product_idea: str, market_context: dict) -> dict:
    """Analyzes competitive landscape and moat potential."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze competitive landscape for: {product_idea}
        Consider:
        1. Existing competitors
        2. Potential competitors
        3. Competitive advantages
        4. Entry barriers
        5. Defensibility
        
        Context: {json.dumps(market_context)}
        """)
    ])
    
    return call_llm(prompt)


def generate_altman_output(product_idea: str, analysis_data: dict) -> SamAltmanSignal:
    """Generates final output using Sam Altman's perspective."""
    # Define the expected output format
    output_format = {
        "opportunity_score": "float between 0 and 1",
        "market_potential": "float between 0 and 1",
        "technical_feasibility": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_insights": "list of strings with key insights",
        "potential_risks": "list of strings with potential risks"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Sam Altman, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. Opportunity score (0-1)
        2. Market potential score (0-1)
        3. Technical feasibility score (0-1)
        4. Detailed reasoning
        5. Key insights
        6. Potential risks
        
        Format your response as a valid JSON object with the following structure:
        {{
            "opportunity_score": <float between 0 and 1>,
            "market_potential": <float between 0 and 1>,
            "technical_feasibility": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_insights": ["<insight 1>", "<insight 2>", ...],
            "potential_risks": ["<risk 1>", "<risk 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Sam Altman evaluation")
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
                        "opportunity_score": 0.5,
                        "market_potential": 0.5,
                        "technical_feasibility": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_insights": ["Default insight"],
                        "potential_risks": ["Default risk"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                # Create a default response
                parsed_response = {
                    "opportunity_score": 0.5,
                    "market_potential": 0.5,
                    "technical_feasibility": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_insights": ["Default insight"],
                    "potential_risks": ["Default risk"]
                }
        else:
            # Response is already a dict
            parsed_response = response
        
        # Ensure all required fields are present
        if "opportunity_score" not in parsed_response:
            parsed_response["opportunity_score"] = 0.5
        if "market_potential" not in parsed_response:
            parsed_response["market_potential"] = 0.5
        if "technical_feasibility" not in parsed_response:
            parsed_response["technical_feasibility"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_insights" not in parsed_response:
            parsed_response["key_insights"] = ["Default insight"]
        if "potential_risks" not in parsed_response:
            parsed_response["potential_risks"] = ["Default risk"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return SamAltmanSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Sam Altman output: {e}", exc_info=True)
        # Return a default response
        return SamAltmanSignal(
            opportunity_score=0.5,
            market_potential=0.5,
            technical_feasibility=0.5,
            reasoning="Error occurred during evaluation.",
            key_insights=["Default insight"],
            potential_risks=["Default risk"]
        ) 