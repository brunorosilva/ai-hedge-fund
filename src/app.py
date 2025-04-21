import streamlit as st
import os
import json
import logging
import time
from orchestrator import ProductOrchestrator
from dotenv import load_dotenv
from utils.ollama_utils import get_available_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('ai-product-evaluator')

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Set page config
st.set_page_config(
    page_title="AI Product Evaluator",
    page_icon="🤖",
    layout="wide"
)
logger.info("Streamlit page config set")

# Initialize orchestrator
@st.cache_resource
def get_orchestrator():
    logger.info("Initializing ProductOrchestrator")
    start_time = time.time()
    orchestrator = ProductOrchestrator()
    logger.info(f"ProductOrchestrator initialized in {time.time() - start_time:.2f} seconds")
    return orchestrator

# Define available agents
AVAILABLE_AGENTS = {
    "Sam Altman": "Focus on AI/ML opportunities and startup scaling",
    "Demis Hassabis": "Deep learning and scientific breakthroughs",
    "Elon Musk": "Disruptive innovation and first principles thinking",
    "Clement Delangue": "NLP and AI applications",
    "Emad Mostaque": "Open source AI and democratization",
    "Sebastian Thrun": "Autonomous systems and education tech",
    "Adam D'Angelo": "Social platforms and AI infrastructure",
    "Daniel Gross": "AI infrastructure and developer tools",
    "Project Advisor": "Final recommendations and project guidance"
}

# Main app
def main():
    logger.info("Starting main application")
    st.title("🤖 AI Product Ideas Evaluator")
    st.markdown("""
    This tool evaluates your product ideas using AI agents inspired by tech leaders.
    Select the agents you want to evaluate your product from the sidebar.
    """)
    
    # Model selection and agent selection in sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Agent selection
        st.subheader("Select Agents")
        st.markdown("Choose which agents will evaluate your product:")
        selected_agents = st.multiselect(
            "Available Agents",
            options=list(AVAILABLE_AGENTS.keys()),
            default=["Sam Altman", "Project Advisor"],
            help="Select one or more agents to evaluate your product"
        )
        
        if not selected_agents:
            st.warning("Please select at least one agent to evaluate your product.")
            return
        
        # Display selected agents and their expertise
        st.markdown("### Selected Agents' Expertise")
        for agent in selected_agents:
            st.markdown(f"**{agent}**: {AVAILABLE_AGENTS[agent]}")
        
        st.divider()
        
        # Model settings
        st.subheader("Model Settings")
        
        # Choose between OpenAI and Ollama
        model_provider = st.radio(
            "Select Model Provider",
            ["OpenAI", "Ollama"],
            index=0 if os.getenv("USE_OLLAMA", "false").lower() != "true" else 1
        )
        logger.info(f"Selected model provider: {model_provider}")
        
        if model_provider == "OpenAI":
            # OpenAI settings
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not found in environment variables")
                st.error("⚠️ OPENAI_API_KEY not found in environment variables.")
                st.info("Create a .env file with your OpenAI API key: OPENAI_API_KEY=your_key_here")
                return
            
            model_name = st.selectbox(
                "OpenAI Model",
                ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                index=0
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                step=0.1
            )
            
            # Update environment variables
            os.environ["USE_OLLAMA"] = "false"
            os.environ["OPENAI_MODEL"] = model_name
            os.environ["OPENAI_TEMPERATURE"] = str(temperature)
            logger.info(f"OpenAI settings updated: model={model_name}, temperature={temperature}")
            
        else:
            # Ollama settings
            try:
                logger.info("Fetching available Ollama models")
                available_models = get_available_models()
                if not available_models:
                    logger.warning("No Ollama models found")
                    st.warning("No Ollama models found. Please make sure Ollama is running.")
                    available_models = ["llama3.1", "mistral", "gemma"]
                
                model_name = st.selectbox(
                    "Ollama Model",
                    available_models,
                    index=0
                )
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1
                )
                
                # Update environment variables
                os.environ["USE_OLLAMA"] = "true"
                os.environ["OLLAMA_MODEL"] = model_name
                logger.info(f"Ollama settings updated: model={model_name}, temperature={temperature}")
                
            except Exception as e:
                logger.error(f"Error connecting to Ollama: {e}", exc_info=True)
                st.error(f"Error connecting to Ollama: {e}")
                st.info("Make sure Ollama is running on your machine.")
                return
    
    # Product idea input
    product_idea = st.text_area(
        "Describe your product idea:",
        height=150,
        placeholder="Enter a detailed description of your product idea..."
    )
    
    # Optional context inputs
    with st.expander("Additional Context (Optional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            market_context = st.text_area(
                "Market Context:",
                height=100,
                placeholder="Describe the market, competitors, target audience..."
            )
        
        with col2:
            technical_context = st.text_area(
                "Technical Context:",
                height=100,
                placeholder="Describe the technical requirements, technologies, challenges..."
            )
    
    # Evaluate button
    if st.button("Evaluate Product Idea", type="primary"):
        if not product_idea:
            logger.warning("Evaluation attempted without product idea")
            st.warning("Please enter a product idea to evaluate.")
            return
        
        if not selected_agents:
            logger.warning("Evaluation attempted without selecting agents")
            st.warning("Please select at least one agent to evaluate your product.")
            return
        
        logger.info("Starting product idea evaluation")
        start_time = time.time()
        
        with st.spinner("Evaluating your product idea..."):
            # Prepare context
            context = {}
            if market_context:
                context["market_context"] = {"description": market_context}
            if technical_context:
                context["technical_context"] = {"description": technical_context}
            
            logger.info(f"Evaluation context prepared: {json.dumps(context)}")
            
            # Get evaluation
            try:
                orchestrator = get_orchestrator()
                logger.info("Calling orchestrator.evaluate_product")
                # Pass selected agents to the orchestrator
                evaluation = orchestrator.evaluate_product(
                    product_idea, 
                    selected_agents=selected_agents,
                    **context
                )
                logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
                
                # Display results
                display_results(evaluation)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)
                st.error(f"An error occurred during evaluation: {e}")

def display_results(evaluation):
    """Display evaluation results in a structured format."""
    logger.info("Displaying evaluation results")
    
    # Overall score
    st.header("Overall Evaluation")
    
    # Score metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Score", f"{evaluation.overall_score:.2f}")
    
    with col2:
        st.metric("Market Potential", f"{evaluation.market_potential:.2f}")
    
    with col3:
        st.metric("Technical Feasibility", f"{evaluation.technical_feasibility:.2f}")
    
    with col4:
        st.metric("Innovation Potential", f"{evaluation.innovation_potential:.2f}")
    
    # Key insights and risks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Insights")
        for insight in evaluation.key_insights:
            st.markdown(f"- {insight}")
    
    with col2:
        st.subheader("Potential Risks")
        for risk in evaluation.potential_risks:
            st.markdown(f"- {risk}")
    
    # Recommendations
    st.subheader("Recommendations")
    for i, rec in enumerate(evaluation.recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Individual agent insights
    st.subheader("Individual Agent Insights")
    
    for agent_name, agent_insight in evaluation.agent_insights.items():
        if agent_insight:
            with st.expander(f"{agent_name.replace('_', ' ').title()} Insights"):
                st.json(agent_insight.dict())
    
    logger.info("Evaluation results displayed successfully")

if __name__ == "__main__":
    logger.info("Application starting")
    main()
    logger.info("Application completed") 