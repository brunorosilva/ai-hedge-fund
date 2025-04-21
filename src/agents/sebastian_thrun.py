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

class SebastianThrunSignal(BaseModel):
    autonomous_systems_score: float  # 0-1 score for autonomous systems
    educational_impact: float  # 0-1 score for educational impact
    innovation_potential: float  # 0-1 score for innovation potential
    reasoning: str
    key_innovations: list[str]
    technical_challenges: list[str]


def sebastian_thrun_agent(state: AgentState):
    """Analyzes product ideas using Sebastian Thrun's expertise in autonomous systems and education."""
    data = state["data"]
    product_idea = data["product_idea"]
    technical_context = data.get("technical_context", {})
    educational_context = data.get("educational_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "autonomous_analysis": analyze_autonomous_systems(product_idea, technical_context),
        "educational_analysis": analyze_educational_impact(product_idea, educational_context),
        "innovation_analysis": analyze_innovation_potential(product_idea),
        "implementation_potential": analyze_implementation_potential(product_idea, technical_context)
    }

    return generate_thrun_output(product_idea, analysis_data)


def analyze_autonomous_systems(product_idea: str, technical_context: dict) -> dict:
    """Analyzes autonomous systems potential and requirements."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze autonomous systems potential for: {product_idea}
        Consider:
        1. Autonomous capabilities
        2. Safety and reliability
        3. Decision-making systems
        4. Sensor integration
        5. System architecture
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_educational_impact(product_idea: str, educational_context: dict) -> dict:
    """Analyzes educational impact and learning potential."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate educational impact for: {product_idea}
        Consider:
        1. Learning outcomes
        2. Educational innovation
        3. Student engagement
        4. Accessibility
        5. Scalability of education
        
        Context: {json.dumps(educational_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_innovation_potential(product_idea: str) -> dict:
    """Analyzes innovation potential and technological advancement."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate innovation potential for: {product_idea}
        Consider:
        1. Technological breakthroughs
        2. Market disruption
        3. Competitive advantages
        4. Future applications
        5. Industry impact
        """)
    ])
    
    return call_llm(prompt)


def analyze_implementation_potential(product_idea: str, technical_context: dict) -> dict:
    """Analyzes implementation potential and technical requirements."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze implementation potential for: {product_idea}
        Consider:
        1. Technical feasibility
        2. Resource requirements
        3. Development timeline
        4. Integration challenges
        5. Maintenance needs
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def generate_thrun_output(product_idea: str, analysis_data: dict) -> SebastianThrunSignal:
    """Generates final output using Sebastian Thrun's perspective."""
    output_format = {
        "autonomous_systems_score": "float between 0 and 1",
        "educational_impact": "float between 0 and 1",
        "innovation_potential": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_innovations": "list of strings with key innovations",
        "technical_challenges": "list of strings with technical challenges"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Sebastian Thrun, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. Autonomous systems score (0-1)
        2. Educational impact score (0-1)
        3. Innovation potential score (0-1)
        4. Detailed reasoning
        5. Key innovations
        6. Technical challenges
        
        Format your response as a valid JSON object with the following structure:
        {{
            "autonomous_systems_score": <float between 0 and 1>,
            "educational_impact": <float between 0 and 1>,
            "innovation_potential": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_innovations": ["<innovation 1>", "<innovation 2>", ...],
            "technical_challenges": ["<challenge 1>", "<challenge 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Sebastian Thrun evaluation")
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
                        "autonomous_systems_score": 0.5,
                        "educational_impact": 0.5,
                        "innovation_potential": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_innovations": ["Default innovation"],
                        "technical_challenges": ["Default challenge"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                parsed_response = {
                    "autonomous_systems_score": 0.5,
                    "educational_impact": 0.5,
                    "innovation_potential": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_innovations": ["Default innovation"],
                    "technical_challenges": ["Default challenge"]
                }
        else:
            parsed_response = response
        
        # Ensure all required fields are present
        if "autonomous_systems_score" not in parsed_response:
            parsed_response["autonomous_systems_score"] = 0.5
        if "educational_impact" not in parsed_response:
            parsed_response["educational_impact"] = 0.5
        if "innovation_potential" not in parsed_response:
            parsed_response["innovation_potential"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_innovations" not in parsed_response:
            parsed_response["key_innovations"] = ["Default innovation"]
        if "technical_challenges" not in parsed_response:
            parsed_response["technical_challenges"] = ["Default challenge"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return SebastianThrunSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Sebastian Thrun output: {e}", exc_info=True)
        return SebastianThrunSignal(
            autonomous_systems_score=0.5,
            educational_impact=0.5,
            innovation_potential=0.5,
            reasoning="Error occurred during evaluation.",
            key_innovations=["Default innovation"],
            technical_challenges=["Default challenge"]
        ) 