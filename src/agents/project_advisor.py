from typing import List, Dict, Any
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from graph.state import AgentState
from utils.llm import call_llm
import json


class ProjectTimeline(BaseModel):
    """Timeline phases for project execution."""
    phase1: str  # Initial setup and planning (1-2 months)
    phase2: str  # Core development (3-6 months)
    phase3: str  # Testing and refinement (2-3 months)
    phase4: str  # Launch preparation (1-2 months)
    total_duration: str  # Total estimated duration
    key_milestones: List[str]  # Key milestones to track


class ProjectRecommendation(BaseModel):
    """Final project recommendation and analysis."""
    pursue_project: bool  # Whether to pursue the project
    confidence_score: float  # Confidence in the recommendation (0-1)
    key_factors: List[str]  # Key factors influencing the decision
    resource_requirements: List[str]  # Required resources and team
    timeline: ProjectTimeline  # Proposed project timeline
    next_steps: List[str]  # Immediate next steps if pursuing
    alternative_suggestions: List[str]  # Alternative suggestions if not pursuing


def project_advisor_agent(state: AgentState) -> ProjectRecommendation:
    """
    Analyzes all agent insights and user background to provide a final recommendation
    on whether to pursue the project and a proposed timeline.
    """
    # Extract data from state
    product_idea = state.data.get("product_idea", "")
    user_background = state.data.get("user_background", {})
    agent_insights = state.data.get("agent_insights", {})
    
    # Collect all insights and scores
    insights_data = _collect_insights_data(agent_insights)
    
    # Generate recommendation using LLM
    recommendation = _generate_recommendation(
        product_idea,
        user_background,
        insights_data
    )
    
    return recommendation


def _collect_insights_data(agent_insights: Dict[str, Any]) -> Dict[str, Any]:
    """Collects and organizes insights from all agents."""
    insights_data = {
        "scores": {},
        "insights": [],
        "risks": [],
        "expertise_areas": set()
    }
    
    # Process each agent's insights
    for agent_name, insights in agent_insights.items():
        if not insights:
            continue
            
        # Add scores
        if hasattr(insights, "opportunity_score"):
            insights_data["scores"]["opportunity"] = insights.opportunity_score
        if hasattr(insights, "market_potential"):
            insights_data["scores"]["market"] = insights.market_potential
        if hasattr(insights, "technical_feasibility"):
            insights_data["scores"]["technical"] = insights.technical_feasibility
            
        # Add insights
        if hasattr(insights, "key_insights"):
            insights_data["insights"].extend(insights.key_insights)
        if hasattr(insights, "key_breakthroughs"):
            insights_data["insights"].extend(insights.key_breakthroughs)
        if hasattr(insights, "key_features"):
            insights_data["insights"].extend(insights.key_features)
            
        # Add risks
        if hasattr(insights, "potential_risks"):
            insights_data["risks"].extend(insights.potential_risks)
        if hasattr(insights, "research_challenges"):
            insights_data["risks"].extend(insights.research_challenges)
            
        # Add expertise areas
        if hasattr(insights, "technical_advancement"):
            insights_data["expertise_areas"].add("technical")
        if hasattr(insights, "ai_infrastructure"):
            insights_data["expertise_areas"].add("ai")
        if hasattr(insights, "platform_potential"):
            insights_data["expertise_areas"].add("platform")
            
    return insights_data


def _generate_recommendation(
    product_idea: str,
    user_background: Dict[str, Any],
    insights_data: Dict[str, Any]
) -> ProjectRecommendation:
    """Generates final project recommendation using LLM."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an experienced project advisor with expertise in evaluating AI and technology projects.
        Your task is to analyze the provided information and make a recommendation on whether to pursue the project.
        Consider the user's background, market potential, technical feasibility, and resource requirements.
        
        You must respond with a valid JSON object that exactly matches this structure:
        {
            "pursue_project": boolean,
            "confidence_score": float (0-1),
            "key_factors": [string],
            "resource_requirements": [string],
            "timeline": {
                "phase1": string,
                "phase2": string,
                "phase3": string,
                "phase4": string,
                "total_duration": string,
                "key_milestones": [string]
            },
            "next_steps": [string],
            "alternative_suggestions": [string]
        }
        
        Guidelines:
        1. If recommending to pursue (pursue_project=true):
           - Provide specific timeline phases with durations
           - List concrete resource requirements
           - Give actionable next steps
        2. If not recommending (pursue_project=false):
           - Explain key factors against pursuing
           - Suggest alternative approaches
           - Provide constructive feedback
        3. Confidence score should reflect certainty in the recommendation
        4. All lists should contain 3-5 specific items
        """),
        ("human", """Product Idea: {product_idea}

User Background:
{user_background}

Expert Analysis:
Scores: {scores}
Key Insights: {insights}
Potential Risks: {risks}
Required Expertise: {expertise}

Based on this information, provide a recommendation on whether to pursue this project.
Remember to respond with a valid JSON object matching the exact structure specified above.""")
    ])
    
    # Format the prompt with the data
    formatted_prompt = prompt.format(
        product_idea=product_idea,
        user_background=json.dumps(user_background, indent=2),
        scores=json.dumps(insights_data["scores"], indent=2),
        insights=json.dumps(insights_data["insights"], indent=2),
        risks=json.dumps(insights_data["risks"], indent=2),
        expertise=json.dumps(list(insights_data["expertise_areas"]), indent=2)
    )
    
    # Get LLM response
    response = call_llm(formatted_prompt)
    
    try:
        response = response.split("```")[1]        
        # Parse the response into a ProjectRecommendation object
        recommendation_data = json.dumps(response)
        
        # Validate required fields
        required_fields = {
            "pursue_project", "confidence_score", "key_factors",
            "resource_requirements", "timeline", "next_steps",
            "alternative_suggestions"
        }
        if not all(field in recommendation_data for field in required_fields):
            raise ValueError("Missing required fields in recommendation")
            
        # Validate timeline fields
        timeline_fields = {
            "phase1", "phase2", "phase3", "phase4",
            "total_duration", "key_milestones"
        }
        if not all(field in recommendation_data["timeline"] for field in timeline_fields):
            raise ValueError("Missing required fields in timeline")
            
        return ProjectRecommendation(**recommendation_data)
    except Exception as e:
        print(f"Error parsing recommendation: {str(e)}")
        print(f"Raw response: {response}")
        
        # Create a more informative fallback recommendation
        return ProjectRecommendation(
            pursue_project=False,
            confidence_score=0.0,
            key_factors=[
                "Error occurred during analysis",
                "Unable to parse recommendation",
                "Please check the input data and try again"
            ],
            resource_requirements=[
                "Technical review needed",
                "Data validation required",
                "System check necessary"
            ],
            timeline=ProjectTimeline(
                phase1="Analysis phase pending",
                phase2="Development phase pending",
                phase3="Testing phase pending",
                phase4="Launch phase pending",
                total_duration="To be determined",
                key_milestones=[
                    "Initial analysis completion",
                    "Technical validation",
                    "Resource assessment"
                ]
            ),
            next_steps=[
                "Review input data for completeness",
                "Validate agent insights",
                "Retry analysis with verified data"
            ],
            alternative_suggestions=[
                "Consider simplifying the product idea",
                "Break down into smaller components",
                "Gather more specific requirements"
            ]
        ) 