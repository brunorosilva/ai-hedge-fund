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

class ElonMuskSignal(BaseModel):
    opportunity_score: float  # 0-1 score for opportunity
    market_potential: float  # 0-1 score for market size
    technical_feasibility: float  # 0-1 score for technical implementation
    reasoning: str
    key_insights: list[str]
    potential_risks: list[str]


def elon_musk_agent(state: AgentState):
    """Analyzes product ideas using Elon Musk's first principles thinking and innovation approach."""
    data = state["data"]
    product_idea = data["product_idea"]
    technical_context = data.get("technical_context", {})
    market_context = data.get("market_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "first_principles": analyze_first_principles(product_idea, technical_context),
        "innovation_analysis": analyze_innovation_potential(product_idea, market_context),
        "execution_analysis": analyze_execution_feasibility(product_idea),
        "disruption_potential": analyze_disruption_potential(product_idea, market_context)
    }

    return generate_musk_output(product_idea, analysis_data)


def analyze_first_principles(product_idea: str, technical_context: dict) -> dict:
    """Analyzes the product from first principles perspective."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze this product idea using first principles thinking: {product_idea}
        Consider:
        1. Fundamental truths and assumptions
        2. Core problems being solved
        3. Novel approaches to solving these problems
        4. Potential for radical innovation
        5. Breaking down complex problems into simpler ones
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_innovation_potential(product_idea: str, market_context: dict) -> dict:
    """Analyzes innovation potential and market disruption."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate innovation potential for: {product_idea}
        Consider:
        1. Market disruption potential
        2. Technological advantages
        3. Competitive moat
        4. Scalability
        5. Network effects
        
        Context: {json.dumps(market_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_execution_feasibility(product_idea: str) -> dict:
    """Analyzes execution feasibility and implementation strategy."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate execution feasibility for: {product_idea}
        Consider:
        1. Resource requirements
        2. Timeline to market
        3. Technical dependencies
        4. Manufacturing challenges
        5. Operational complexity
        """)
    ])
    
    return call_llm(prompt)


def analyze_disruption_potential(product_idea: str, market_context: dict) -> dict:
    """Analyzes potential for market disruption and transformation."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze disruption potential for: {product_idea}
        Consider:
        1. Market transformation potential
        2. Industry impact
        3. Competitive advantages
        4. Market size and growth
        5. Long-term vision
        
        Context: {json.dumps(market_context)}
        """)
    ])
    
    return call_llm(prompt)


def generate_musk_output(product_idea: str, analysis_data: dict) -> ElonMuskSignal:
    """Generates final output using Elon Musk's perspective."""
    output_format = {
        "first_principles_score": "float between 0 and 1",
        "innovation_potential": "float between 0 and 1",
        "execution_feasibility": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_innovations": "list of strings with key innovations",
        "technical_challenges": "list of strings with technical challenges"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Elon Musk, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. First principles thinking score (0-1)
        2. Innovation potential score (0-1)
        3. Execution feasibility score (0-1)
        4. Detailed reasoning
        5. Key innovations
        6. Technical challenges
        
        Format your response as a valid JSON object with the following structure:
        {{
            "first_principles_score": <float between 0 and 1>,
            "innovation_potential": <float between 0 and 1>,
            "execution_feasibility": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_innovations": ["<innovation 1>", "<innovation 2>", ...],
            "technical_challenges": ["<challenge 1>", "<challenge 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Elon Musk evaluation")
        response = call_llm(prompt, output_format=str(output_format))
        logger.info(f"Raw response from LLM: {response}")
        
        # If response is a string, try to parse it as JSON
        if isinstance(response, str):
            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    logger.warning("No JSON found in response, creating default")
                    parsed_response = {
                        "first_principles_score": 0.5,
                        "innovation_potential": 0.5,
                        "execution_feasibility": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_innovations": ["Default innovation"],
                        "technical_challenges": ["Default challenge"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                parsed_response = {
                    "first_principles_score": 0.5,
                    "innovation_potential": 0.5,
                    "execution_feasibility": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_innovations": ["Default innovation"],
                    "technical_challenges": ["Default challenge"]
                }
        else:
            parsed_response = response
        
        # Ensure all required fields are present
        if "first_principles_score" not in parsed_response:
            parsed_response["first_principles_score"] = 0.5
        if "innovation_potential" not in parsed_response:
            parsed_response["innovation_potential"] = 0.5
        if "execution_feasibility" not in parsed_response:
            parsed_response["execution_feasibility"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_innovations" not in parsed_response:
            parsed_response["key_innovations"] = ["Default innovation"]
        if "technical_challenges" not in parsed_response:
            parsed_response["technical_challenges"] = ["Default challenge"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return ElonMuskSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Elon Musk output: {e}", exc_info=True)
        return ElonMuskSignal(
            first_principles_score=0.5,
            innovation_potential=0.5,
            execution_feasibility=0.5,
            reasoning="Error occurred during evaluation.",
            key_innovations=["Default innovation"],
            technical_challenges=["Default challenge"]
        ) 