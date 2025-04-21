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

class ClementDelangueSignal(BaseModel):
    ai_innovation_score: float  # 0-1 score for AI/ML innovation
    practical_application: float  # 0-1 score for practical application
    technical_feasibility: float  # 0-1 score for technical feasibility
    reasoning: str
    key_ai_features: list[str]
    implementation_challenges: list[str]


def clement_delangue_agent(state: AgentState):
    """Analyzes product ideas using Clement Delangue's AI/ML expertise and practical approach."""
    data = state["data"]
    product_idea = data["product_idea"]
    technical_context = data.get("technical_context", {})
    ai_context = data.get("ai_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "ai_analysis": analyze_ai_innovation(product_idea, ai_context),
        "practical_analysis": analyze_practical_application(product_idea, technical_context),
        "technical_analysis": analyze_technical_feasibility(product_idea),
        "implementation_potential": analyze_implementation_potential(product_idea, technical_context)
    }

    return generate_delangue_output(product_idea, analysis_data)


def analyze_ai_innovation(product_idea: str, ai_context: dict) -> dict:
    """Analyzes AI/ML innovation potential."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze AI/ML innovation potential for: {product_idea}
        Consider:
        1. Novel AI/ML approaches
        2. State-of-the-art techniques
        3. Model architecture innovation
        4. Data requirements
        5. AI/ML competitive advantages
        
        Context: {json.dumps(ai_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_practical_application(product_idea: str, technical_context: dict) -> dict:
    """Analyzes practical application and real-world impact."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate practical application for: {product_idea}
        Consider:
        1. Real-world use cases
        2. User value proposition
        3. Market fit
        4. Scalability
        5. Integration potential
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def analyze_technical_feasibility(product_idea: str) -> dict:
    """Analyzes technical feasibility and implementation requirements."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate technical feasibility for: {product_idea}
        Consider:
        1. AI/ML infrastructure needs
        2. Data pipeline requirements
        3. Model deployment complexity
        4. Performance considerations
        5. Maintenance requirements
        """)
    ])
    
    return call_llm(prompt)


def analyze_implementation_potential(product_idea: str, technical_context: dict) -> dict:
    """Analyzes implementation potential and technical roadmap."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze implementation potential for: {product_idea}
        Consider:
        1. Development timeline
        2. Resource requirements
        3. Technical dependencies
        4. Integration challenges
        5. Scaling considerations
        
        Context: {json.dumps(technical_context)}
        """)
    ])
    
    return call_llm(prompt)


def generate_delangue_output(product_idea: str, analysis_data: dict) -> ClementDelangueSignal:
    """Generates final output using Clement Delangue's perspective."""
    output_format = {
        "ai_innovation_score": "float between 0 and 1",
        "practical_application": "float between 0 and 1",
        "technical_feasibility": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_ai_features": "list of strings with key AI features",
        "implementation_challenges": "list of strings with implementation challenges"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Clement Delangue, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. AI/ML innovation score (0-1)
        2. Practical application score (0-1)
        3. Technical feasibility score (0-1)
        4. Detailed reasoning
        5. Key AI features
        6. Implementation challenges
        
        Format your response as a valid JSON object with the following structure:
        {{
            "ai_innovation_score": <float between 0 and 1>,
            "practical_application": <float between 0 and 1>,
            "technical_feasibility": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_ai_features": ["<feature 1>", "<feature 2>", ...],
            "implementation_challenges": ["<challenge 1>", "<challenge 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Clement Delangue evaluation")
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
                        "ai_innovation_score": 0.5,
                        "practical_application": 0.5,
                        "technical_feasibility": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_ai_features": ["Default feature"],
                        "implementation_challenges": ["Default challenge"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                parsed_response = {
                    "ai_innovation_score": 0.5,
                    "practical_application": 0.5,
                    "technical_feasibility": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_ai_features": ["Default feature"],
                    "implementation_challenges": ["Default challenge"]
                }
        else:
            parsed_response = response
        
        # Ensure all required fields are present
        if "ai_innovation_score" not in parsed_response:
            parsed_response["ai_innovation_score"] = 0.5
        if "practical_application" not in parsed_response:
            parsed_response["practical_application"] = 0.5
        if "technical_feasibility" not in parsed_response:
            parsed_response["technical_feasibility"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_ai_features" not in parsed_response:
            parsed_response["key_ai_features"] = ["Default feature"]
        if "implementation_challenges" not in parsed_response:
            parsed_response["implementation_challenges"] = ["Default challenge"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return ClementDelangueSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Clement Delangue output: {e}", exc_info=True)
        return ClementDelangueSignal(
            ai_innovation_score=0.5,
            practical_application=0.5,
            technical_feasibility=0.5,
            reasoning="Error occurred during evaluation.",
            key_ai_features=["Default feature"],
            implementation_challenges=["Default challenge"]
        ) 