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

class EmadMostaqueSignal(BaseModel):
    infrastructure_score: float  # 0-1 score for AI infrastructure
    open_source_potential: float  # 0-1 score for open source potential
    community_impact: float  # 0-1 score for community impact
    reasoning: str
    key_infrastructure: list[str]
    community_challenges: list[str]


def emad_mostaque_agent(state: AgentState):
    """Analyzes product ideas using Emad Mostaque's expertise in AI infrastructure and open source."""
    data = state["data"]
    product_idea = data["product_idea"]
    technical_context = data.get("technical_context", {})
    community_context = data.get("community_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "infrastructure_analysis": analyze_infrastructure_potential(product_idea, technical_context),
        "open_source_analysis": analyze_open_source_potential(product_idea, community_context),
        "community_analysis": analyze_community_impact(product_idea),
        "scaling_potential": analyze_scaling_potential(product_idea, technical_context)
    }

    return generate_mostaque_output(product_idea, analysis_data)


def analyze_infrastructure_potential(product_idea: str, technical_context: dict) -> dict:
    """Analyzes AI infrastructure potential and requirements."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze infrastructure potential for: {product_idea}
        Consider:
        1. AI infrastructure requirements
        2. Scalability needs
        3. Resource optimization
        4. Technical architecture
        5. Infrastructure challenges
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_open_source_potential(product_idea: str, community_context: dict) -> dict:
    """Analyzes open source potential and community aspects."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate open source potential for: {product_idea}
        Consider:
        1. Community engagement potential
        2. Open source licensing
        3. Contribution opportunities
        4. Documentation needs
        5. Community governance
        
        Context: {json.dumps(community_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_community_impact(product_idea: str) -> dict:
    """Analyzes potential impact on the AI community."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate community impact for: {product_idea}
        Consider:
        1. Developer adoption potential
        2. Community value proposition
        3. Knowledge sharing opportunities
        4. Collaboration potential
        5. Long-term community growth
        """)
    ])
    
    return call_llm(prompt)


def analyze_scaling_potential(product_idea: str, technical_context: dict) -> dict:
    """Analyzes scaling potential and infrastructure requirements."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze scaling potential for: {product_idea}
        Consider:
        1. Infrastructure scaling needs
        2. Performance optimization
        3. Resource management
        4. Cost considerations
        5. Technical limitations
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def generate_mostaque_output(product_idea: str, analysis_data: dict) -> EmadMostaqueSignal:
    """Generates final output using Emad Mostaque's perspective."""
    output_format = {
        "infrastructure_score": "float between 0 and 1",
        "open_source_potential": "float between 0 and 1",
        "community_impact": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_infrastructure": "list of strings with key infrastructure components",
        "community_challenges": "list of strings with community challenges"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Emad Mostaque, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. Infrastructure score (0-1)
        2. Open source potential score (0-1)
        3. Community impact score (0-1)
        4. Detailed reasoning
        5. Key infrastructure components
        6. Community challenges
        
        Format your response as a valid JSON object with the following structure:
        {{
            "infrastructure_score": <float between 0 and 1>,
            "open_source_potential": <float between 0 and 1>,
            "community_impact": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_infrastructure": ["<component 1>", "<component 2>", ...],
            "community_challenges": ["<challenge 1>", "<challenge 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Emad Mostaque evaluation")
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
                        "infrastructure_score": 0.5,
                        "open_source_potential": 0.5,
                        "community_impact": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_infrastructure": ["Default component"],
                        "community_challenges": ["Default challenge"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                parsed_response = {
                    "infrastructure_score": 0.5,
                    "open_source_potential": 0.5,
                    "community_impact": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_infrastructure": ["Default component"],
                    "community_challenges": ["Default challenge"]
                }
        else:
            parsed_response = response
        
        # Ensure all required fields are present
        if "infrastructure_score" not in parsed_response:
            parsed_response["infrastructure_score"] = 0.5
        if "open_source_potential" not in parsed_response:
            parsed_response["open_source_potential"] = 0.5
        if "community_impact" not in parsed_response:
            parsed_response["community_impact"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_infrastructure" not in parsed_response:
            parsed_response["key_infrastructure"] = ["Default component"]
        if "community_challenges" not in parsed_response:
            parsed_response["community_challenges"] = ["Default challenge"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return EmadMostaqueSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Emad Mostaque output: {e}", exc_info=True)
        return EmadMostaqueSignal(
            infrastructure_score=0.5,
            open_source_potential=0.5,
            community_impact=0.5,
            reasoning="Error occurred during evaluation.",
            key_infrastructure=["Default component"],
            community_challenges=["Default challenge"]
        ) 