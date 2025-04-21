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

class AdamDAngeloSignal(BaseModel):
    platform_potential: float  # 0-1 score for platform potential
    ai_infrastructure: float  # 0-1 score for AI infrastructure
    social_impact: float  # 0-1 score for social impact
    reasoning: str
    key_features: list[str]
    platform_challenges: list[str]


def adam_dangelo_agent(state: AgentState):
    """Analyzes product ideas using Adam D'Angelo's expertise in AI infrastructure and social platforms."""
    data = state["data"]
    product_idea = data["product_idea"]
    technical_context = data.get("technical_context", {})
    social_context = data.get("social_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "platform_analysis": analyze_platform_potential(product_idea, social_context),
        "infrastructure_analysis": analyze_ai_infrastructure(product_idea, technical_context),
        "social_analysis": analyze_social_impact(product_idea),
        "scaling_potential": analyze_scaling_potential(product_idea, technical_context)
    }

    return generate_dangelo_output(product_idea, analysis_data)


def analyze_platform_potential(product_idea: str, social_context: dict) -> dict:
    """Analyzes platform potential and social dynamics."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze platform potential for: {product_idea}
        Consider:
        1. Network effects
        2. User engagement
        3. Community building
        4. Platform stickiness
        5. Growth potential
        
        Context: {json.dumps(social_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_ai_infrastructure(product_idea: str, technical_context: dict) -> dict:
    """Analyzes AI infrastructure and technical requirements."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate AI infrastructure for: {product_idea}
        Consider:
        1. AI/ML architecture
        2. Data pipeline
        3. Model deployment
        4. Performance optimization
        5. Infrastructure scaling
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_social_impact(product_idea: str) -> dict:
    """Analyzes social impact and community aspects."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate social impact for: {product_idea}
        Consider:
        1. Community value
        2. Social dynamics
        3. User experience
        4. Privacy considerations
        5. Ethical implications
        """)
    ])
    
    return call_llm(prompt)


def analyze_scaling_potential(product_idea: str, technical_context: dict) -> dict:
    """Analyzes scaling potential and technical requirements."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze scaling potential for: {product_idea}
        Consider:
        1. Infrastructure scaling
        2. Performance at scale
        3. Cost efficiency
        4. Technical limitations
        5. Maintenance requirements
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def generate_dangelo_output(product_idea: str, analysis_data: dict) -> AdamDAngeloSignal:
    """Generates final output using Adam D'Angelo's perspective."""
    output_format = {
        "platform_potential": "float between 0 and 1",
        "ai_infrastructure": "float between 0 and 1",
        "social_impact": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_features": "list of strings with key features",
        "platform_challenges": "list of strings with platform challenges"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Adam D'Angelo, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. Platform potential score (0-1)
        2. AI infrastructure score (0-1)
        3. Social impact score (0-1)
        4. Detailed reasoning
        5. Key features
        6. Platform challenges
        
        Format your response as a valid JSON object with the following structure:
        {{
            "platform_potential": <float between 0 and 1>,
            "ai_infrastructure": <float between 0 and 1>,
            "social_impact": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_features": ["<feature 1>", "<feature 2>", ...],
            "platform_challenges": ["<challenge 1>", "<challenge 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Adam D'Angelo evaluation")
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
                        "platform_potential": 0.5,
                        "ai_infrastructure": 0.5,
                        "social_impact": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_features": ["Default feature"],
                        "platform_challenges": ["Default challenge"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                parsed_response = {
                    "platform_potential": 0.5,
                    "ai_infrastructure": 0.5,
                    "social_impact": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_features": ["Default feature"],
                    "platform_challenges": ["Default challenge"]
                }
        else:
            parsed_response = response
        
        # Ensure all required fields are present
        if "platform_potential" not in parsed_response:
            parsed_response["platform_potential"] = 0.5
        if "ai_infrastructure" not in parsed_response:
            parsed_response["ai_infrastructure"] = 0.5
        if "social_impact" not in parsed_response:
            parsed_response["social_impact"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_features" not in parsed_response:
            parsed_response["key_features"] = ["Default feature"]
        if "platform_challenges" not in parsed_response:
            parsed_response["platform_challenges"] = ["Default challenge"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return AdamDAngeloSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Adam D'Angelo output: {e}", exc_info=True)
        return AdamDAngeloSignal(
            platform_potential=0.5,
            ai_infrastructure=0.5,
            social_impact=0.5,
            reasoning="Error occurred during evaluation.",
            key_features=["Default feature"],
            platform_challenges=["Default challenge"]
        ) 