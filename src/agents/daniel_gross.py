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

class DanielGrossSignal(BaseModel):
    startup_potential: float  # 0-1 score for startup potential
    ai_infrastructure: float  # 0-1 score for AI infrastructure
    market_fit: float  # 0-1 score for market fit
    reasoning: str
    key_advantages: list[str]
    startup_challenges: list[str]


def daniel_gross_agent(state: AgentState):
    """Analyzes product ideas using Daniel Gross's expertise in AI infrastructure and startup development."""
    data = state["data"]
    product_idea = data["product_idea"]
    technical_context = data.get("technical_context", {})
    market_context = data.get("market_context", {})

    # Collect analysis for LLM reasoning
    analysis_data = {
        "startup_analysis": analyze_startup_potential(product_idea, market_context),
        "infrastructure_analysis": analyze_ai_infrastructure(product_idea, technical_context),
        "market_analysis": analyze_market_fit(product_idea),
        "scaling_potential": analyze_scaling_potential(product_idea, technical_context)
    }

    return generate_gross_output(product_idea, analysis_data)


def analyze_startup_potential(product_idea: str, market_context: dict) -> dict:
    """Analyzes startup potential and market opportunity."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Analyze startup potential for: {product_idea}
        Consider:
        1. Market opportunity
        2. Competitive advantages
        3. Growth potential
        4. Resource requirements
        5. Exit potential
        
        Context: {json.dumps(market_context)}
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


def analyze_market_fit(product_idea: str) -> dict:
    """Analyzes market fit and customer needs."""
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""Evaluate market fit for: {product_idea}
        Consider:
        1. Customer needs
        2. Market size
        3. Competition
        4. Pricing strategy
        5. Customer acquisition
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


def generate_gross_output(product_idea: str, analysis_data: dict) -> DanielGrossSignal:
    """Generates final output using Daniel Gross's perspective."""
    output_format = {
        "startup_potential": "float between 0 and 1",
        "ai_infrastructure": "float between 0 and 1",
        "market_fit": "float between 0 and 1",
        "reasoning": "string with detailed reasoning",
        "key_advantages": "list of strings with key advantages",
        "startup_challenges": "list of strings with startup challenges"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=f"""As Daniel Gross, evaluate this product idea: {product_idea}
        
        Analysis data: {json.dumps(analysis_data)}
        
        Provide:
        1. Startup potential score (0-1)
        2. AI infrastructure score (0-1)
        3. Market fit score (0-1)
        4. Detailed reasoning
        5. Key advantages
        6. Startup challenges
        
        Format your response as a valid JSON object with the following structure:
        {{
            "startup_potential": <float between 0 and 1>,
            "ai_infrastructure": <float between 0 and 1>,
            "market_fit": <float between 0 and 1>,
            "reasoning": "<detailed reasoning as a string>",
            "key_advantages": ["<advantage 1>", "<advantage 2>", ...],
            "startup_challenges": ["<challenge 1>", "<challenge 2>", ...]
        }}
        
        Make sure your response is a valid JSON object that can be parsed directly.
        """)
    ])
    
    try:
        logger.info("Calling LLM for Daniel Gross evaluation")
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
                        "startup_potential": 0.5,
                        "ai_infrastructure": 0.5,
                        "market_fit": 0.5,
                        "reasoning": "Unable to generate detailed reasoning.",
                        "key_advantages": ["Default advantage"],
                        "startup_challenges": ["Default challenge"]
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from response: {e}")
                parsed_response = {
                    "startup_potential": 0.5,
                    "ai_infrastructure": 0.5,
                    "market_fit": 0.5,
                    "reasoning": "Unable to generate detailed reasoning.",
                    "key_advantages": ["Default advantage"],
                    "startup_challenges": ["Default challenge"]
                }
        else:
            parsed_response = response
        
        # Ensure all required fields are present
        if "startup_potential" not in parsed_response:
            parsed_response["startup_potential"] = 0.5
        if "ai_infrastructure" not in parsed_response:
            parsed_response["ai_infrastructure"] = 0.5
        if "market_fit" not in parsed_response:
            parsed_response["market_fit"] = 0.5
        if "reasoning" not in parsed_response:
            parsed_response["reasoning"] = "Unable to generate detailed reasoning."
        if "key_advantages" not in parsed_response:
            parsed_response["key_advantages"] = ["Default advantage"]
        if "startup_challenges" not in parsed_response:
            parsed_response["startup_challenges"] = ["Default challenge"]
        
        logger.info(f"Parsed response: {parsed_response}")
        return DanielGrossSignal(**parsed_response)
    
    except Exception as e:
        logger.error(f"Error generating Daniel Gross output: {e}", exc_info=True)
        return DanielGrossSignal(
            startup_potential=0.5,
            ai_infrastructure=0.5,
            market_fit=0.5,
            reasoning="Error occurred during evaluation.",
            key_advantages=["Default advantage"],
            startup_challenges=["Default challenge"]
        ) 