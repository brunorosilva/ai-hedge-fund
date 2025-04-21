from typing import Dict, List, Any, Optional
import json
from pydantic import BaseModel
from graph.state import AgentState
from agents.sam_altman import sam_altman_agent, SamAltmanSignal
from agents.demis_hassabis import demis_hassabis_agent, DemisHassabisSignal
from agents.elon_musk import elon_musk_agent, ElonMuskSignal
from agents.adam_dangelo import adam_dangelo_agent, AdamDAngeloSignal
from agents.daniel_gross import daniel_gross_agent, DanielGrossSignal
from agents.sebastian_thrun import sebastian_thrun_agent, SebastianThrunSignal
from agents.emad_mostaque import emad_mostaque_agent, EmadMostaqueSignal
from agents.clement_delangue import clement_delangue_agent, ClementDelangueSignal
from agents.project_advisor import project_advisor_agent, ProjectRecommendation
from agent_selector import AgentSelector
from utils.llm import call_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


class ProductEvaluation(BaseModel):
    """Unified product evaluation combining insights from all agents."""
    product_idea: str
    overall_score: float  # 0-1 score for overall potential
    market_potential: float  # 0-1 score for market potential
    technical_feasibility: float  # 0-1 score for technical feasibility
    innovation_potential: float  # 0-1 score for innovation potential
    key_insights: List[str]
    potential_risks: List[str]
    agent_insights: Dict[str, Any]  # Individual agent insights
    recommendations: List[str]
    project_recommendation: Optional[ProjectRecommendation] = None  # Final project recommendation


class ProductOrchestrator:
    """Orchestrates product evaluation across multiple AI agents."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config_file: Optional path to the agent configuration file
        """
        # Initialize the agent selector
        self.agent_selector = AgentSelector(config_file)
        
        # Define all available agent functions
        self.all_agents = {
            "Sam Altman": sam_altman_agent,
            "Demis Hassabis": demis_hassabis_agent,
            "Elon Musk": elon_musk_agent,
            "Adam D'Angelo": adam_dangelo_agent,
            "Daniel Gross": daniel_gross_agent,
            "Sebastian Thrun": sebastian_thrun_agent,
            "Emad Mostaque": emad_mostaque_agent,
            "Clement Delangue": clement_delangue_agent,
            "Project Advisor": project_advisor_agent,
        }
    
    def configure_agents(self) -> None:
        """Run the interactive agent configuration process."""
        self.agent_selector.interactive_selection()
    
    def evaluate_product(
        self, 
        product_idea: str, 
        selected_agents: Optional[List[str]] = None,
        market_context: Optional[Dict] = None, 
        technical_context: Optional[Dict] = None,
        user_background: Optional[Dict] = None
    ) -> ProductEvaluation:
        """
        Evaluates a product idea using selected agents and combines their insights.
        
        Args:
            product_idea: The product idea to evaluate
            selected_agents: List of agent names to use for evaluation. If None, uses all agents.
            market_context: Optional market context information
            technical_context: Optional technical context information
            user_background: Optional information about the user's background and experience
            
        Returns:
            ProductEvaluation: Combined evaluation from all agents
        """
        # Get enabled agent functions based on selection
        if selected_agents is None:
            # Use all agents if none specified
            enabled_agents = self.all_agents
        else:
            # Filter agents based on selection
            enabled_agents = {
                name: func for name, func in self.all_agents.items()
                if name in selected_agents
            }
        
        # Prepare state for agents
        state = AgentState({
            "data": {
                "product_idea": product_idea,
                "market_context": market_context or {},
                "technical_context": technical_context or {},
                "user_background": user_background or {}
            }
        })
        
        # Collect insights from enabled agents
        agent_insights = {}
        for agent_name, agent_func in enabled_agents.items():
            try:
                agent_insights[agent_name] = agent_func(state)
            except Exception as e:
                print(f"Error running {agent_name} agent: {e}")
                agent_insights[agent_name] = None
        
        # Combine insights into a unified evaluation
        evaluation = self._combine_insights(product_idea, agent_insights)
        
        # Add project recommendation if available
        if "Project Advisor" in agent_insights and agent_insights["Project Advisor"]:
            evaluation.project_recommendation = agent_insights["Project Advisor"]
        
        return evaluation
    
    def _combine_insights(self, product_idea: str, agent_insights: Dict[str, Any]) -> ProductEvaluation:
        """Combines insights from all agents into a unified evaluation."""
        
        # Extract scores from agent insights
        scores = {
            "opportunity_score": 0.0,
            "market_potential": 0.0,
            "technical_feasibility": 0.0,
            "scientific_breakthrough_potential": 0.0,
            "technical_advancement": 0.0,
            "research_feasibility": 0.0,
            "startup_potential": 0.0,
            "ai_infrastructure": 0.0,
            "platform_potential": 0.0,
            "social_impact": 0.0,
            "autonomous_systems_score": 0.0,
            "educational_impact": 0.0,
            "innovation_potential": 0.0
        }
        
        # Count valid scores for averaging
        score_counts = {k: 0 for k in scores.keys()}
        
        # Process Sam Altman's insights
        if isinstance(agent_insights.get("Sam Altman"), SamAltmanSignal):
            altman = agent_insights["Sam Altman"]
            scores["opportunity_score"] += altman.opportunity_score
            scores["market_potential"] += altman.market_potential
            scores["technical_feasibility"] += altman.technical_feasibility
            score_counts["opportunity_score"] += 1
            score_counts["market_potential"] += 1
            score_counts["technical_feasibility"] += 1
        
        # Process Demis Hassabis's insights
        if isinstance(agent_insights.get("Demis Hassabis"), DemisHassabisSignal):
            hassabis = agent_insights["Demis Hassabis"]
            scores["scientific_breakthrough_potential"] += hassabis.scientific_breakthrough_potential
            scores["technical_advancement"] += hassabis.technical_advancement
            scores["research_feasibility"] += hassabis.research_feasibility
            score_counts["scientific_breakthrough_potential"] += 1
            score_counts["technical_advancement"] += 1
            score_counts["research_feasibility"] += 1
        
        # Process Elon Musk's insights
        if isinstance(agent_insights.get("Elon Musk"), ElonMuskSignal):
            musk = agent_insights["Elon Musk"]
            scores["opportunity_score"] += musk.opportunity_score
            scores["market_potential"] += musk.market_potential
            scores["technical_feasibility"] += musk.technical_feasibility
            score_counts["opportunity_score"] += 1
            score_counts["market_potential"] += 1
            score_counts["technical_feasibility"] += 1
        
        # Process Adam D'Angelo's insights
        if isinstance(agent_insights.get("Adam D'Angelo"), AdamDAngeloSignal):
            dangelo = agent_insights["Adam D'Angelo"]
            scores["platform_potential"] += dangelo.platform_potential
            scores["ai_infrastructure"] += dangelo.ai_infrastructure
            scores["social_impact"] += dangelo.social_impact
            score_counts["platform_potential"] += 1
            score_counts["ai_infrastructure"] += 1
            score_counts["social_impact"] += 1
        
        # Process Daniel Gross's insights
        if isinstance(agent_insights.get("Daniel Gross"), DanielGrossSignal):
            gross = agent_insights["Daniel Gross"]
            scores["startup_potential"] += gross.startup_potential
            scores["ai_infrastructure"] += gross.ai_infrastructure
            scores["market_potential"] += gross.market_fit
            score_counts["startup_potential"] += 1
            score_counts["ai_infrastructure"] += 1
            score_counts["market_potential"] += 1
            
        # Process Sebastian Thrun's insights
        if isinstance(agent_insights.get("Sebastian Thrun"), SebastianThrunSignal):
            thrun = agent_insights["Sebastian Thrun"]
            scores["autonomous_systems_score"] += thrun.autonomous_systems_score
            scores["educational_impact"] += thrun.educational_impact
            scores["innovation_potential"] += thrun.innovation_potential
            score_counts["autonomous_systems_score"] += 1
            score_counts["educational_impact"] += 1
            score_counts["innovation_potential"] += 1
            
        # Process Emad Mostaque's insights
        if isinstance(agent_insights.get("Emad Mostaque"), EmadMostaqueSignal):
            mostaque = agent_insights["Emad Mostaque"]
            scores["ai_infrastructure"] += mostaque.infrastructure_score
            scores["social_impact"] += mostaque.community_impact
            scores["technical_feasibility"] += mostaque.open_source_potential
            score_counts["ai_infrastructure"] += 1
            score_counts["social_impact"] += 1
            score_counts["technical_feasibility"] += 1
            
        # Process Clement Delangue's insights
        if isinstance(agent_insights.get("Clement Delangue"), ClementDelangueSignal):
            delangue = agent_insights["Clement Delangue"]
            scores["ai_infrastructure"] += delangue.ai_innovation_score
            scores["technical_feasibility"] += delangue.technical_feasibility
            scores["market_potential"] += delangue.practical_application
            score_counts["ai_infrastructure"] += 1
            score_counts["technical_feasibility"] += 1
            score_counts["market_potential"] += 1
        
        # Calculate averages
        for key in scores:
            if score_counts[key] > 0:
                scores[key] /= score_counts[key]
        
        # Calculate overall score (weighted average of key metrics)
        key_metrics = [
            "opportunity_score",
            "market_potential",
            "technical_feasibility",
            "scientific_breakthrough_potential",
            "ai_infrastructure",
            "innovation_potential"
        ]
        overall_score = sum(scores[metric] for metric in key_metrics) / len(key_metrics)
        
        # Extract key insights and risks
        key_insights = []
        potential_risks = []
        recommendations = []
        
        # Collect insights from each agent
        if isinstance(agent_insights.get("Sam Altman"), SamAltmanSignal):
            key_insights.extend(agent_insights["Sam Altman"].key_insights)
            potential_risks.extend(agent_insights["Sam Altman"].potential_risks)
        
        if isinstance(agent_insights.get("Demis Hassabis"), DemisHassabisSignal):
            key_insights.extend(agent_insights["Demis Hassabis"].key_breakthroughs)
            potential_risks.extend(agent_insights["Demis Hassabis"].research_challenges)
            
        if isinstance(agent_insights.get("Elon Musk"), ElonMuskSignal):
            key_insights.extend(agent_insights["Elon Musk"].key_insights)
            potential_risks.extend(agent_insights["Elon Musk"].potential_risks)
            
        if isinstance(agent_insights.get("Adam D'Angelo"), AdamDAngeloSignal):
            key_insights.extend(agent_insights["Adam D'Angelo"].key_features)
            potential_risks.extend(agent_insights["Adam D'Angelo"].platform_challenges)
            
        if isinstance(agent_insights.get("Daniel Gross"), DanielGrossSignal):
            key_insights.extend(agent_insights["Daniel Gross"].key_advantages)
            potential_risks.extend(agent_insights["Daniel Gross"].startup_challenges)
            
        if isinstance(agent_insights.get("Sebastian Thrun"), SebastianThrunSignal):
            key_insights.extend(agent_insights["Sebastian Thrun"].key_innovations)
            potential_risks.extend(agent_insights["Sebastian Thrun"].technical_challenges)
            
        if isinstance(agent_insights.get("Emad Mostaque"), EmadMostaqueSignal):
            key_insights.extend(agent_insights["Emad Mostaque"].key_infrastructure)
            potential_risks.extend(agent_insights["Emad Mostaque"].community_challenges)
            
        if isinstance(agent_insights.get("Clement Delangue"), ClementDelangueSignal):
            key_insights.extend(agent_insights["Clement Delangue"].key_ai_features)
            potential_risks.extend(agent_insights["Clement Delangue"].implementation_challenges)
        
        # Process Project Advisor's recommendations
        project_recommendation = None
        if isinstance(agent_insights.get("Project Advisor"), ProjectRecommendation):
            project_recommendation = agent_insights["Project Advisor"]
            # Add Project Advisor's recommendations to the main recommendations list
            recommendations.extend(project_recommendation.recommendations)
        
        # If no recommendations from Project Advisor, generate them using LLM
        if not recommendations:
            recommendations = self._generate_recommendations(
                product_idea, 
                scores, 
                key_insights, 
                potential_risks
            )
        
        return ProductEvaluation(
            product_idea=product_idea,
            overall_score=overall_score,
            market_potential=scores["market_potential"],
            technical_feasibility=scores["technical_feasibility"],
            innovation_potential=scores["scientific_breakthrough_potential"],
            key_insights=key_insights,
            potential_risks=potential_risks,
            agent_insights=agent_insights,
            recommendations=recommendations,
            project_recommendation=project_recommendation
        )
    
    def _generate_recommendations(self, product_idea: str, scores: Dict[str, float], 
                                 key_insights: List[str], potential_risks: List[str]) -> List[str]:
        """Generates recommendations based on the combined insights."""
        prompt = ChatPromptTemplate.from_messages([
            HumanMessage(content="""Based on the following product idea and analysis, provide 5 specific recommendations for moving forward.
Your response must be a valid JSON object of strings.

Product Idea: {product_idea}

Scores:
- Overall Score: {overall_score:.2f}
- Market Potential: {market_potential:.2f}
- Technical Feasibility: {technical_feasibility:.2f}
- Innovation Potential: {scientific_breakthrough_potential:.2f}

Key Insights:
{key_insights}

Potential Risks:
{potential_risks}

Return a JSON object with a key "recommendations" and an array of 5 strings containing specific, actionable recommendations.
""".format(product_idea=product_idea, overall_score=scores.get('opportunity_score', 0),
           market_potential=scores.get('market_potential', 0), technical_feasibility=scores.get('technical_feasibility', 0),
           scientific_breakthrough_potential=scores.get('scientific_breakthrough_potential', 0),
           key_insights=key_insights, potential_risks=potential_risks))
        ])
        
        try:
            response = call_llm(prompt)
            response = response.split("```")[1]
            # Try to parse the response as JSON
            recommendations = json.dumps(response)
            if isinstance(recommendations, dict) and "recommendations" in recommendations:
                return recommendations["recommendations"][:5]  # Ensure we only return up to 5 recommendations
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Raw response: {response}")
        
        # Fallback recommendations if JSON parsing fails
        return [
            "Conduct more detailed market research",
            "Develop a minimum viable product (MVP)",
            "Assemble a diverse team with complementary skills",
            "Create a detailed roadmap with milestones",
            "Establish key performance indicators (KPIs)"
        ]
    