# AI Product Ideas Evaluator

A tool that evaluates product ideas using AI agents inspired by tech leaders like Sam Altman and Demis Hassabis.

## Features

- Multiple AI agents with different perspectives and expertise
- Unified evaluation combining insights from all agents
- Interactive web interface built with Streamlit
- Support for both OpenAI and Ollama models
- Detailed analysis including market potential, technical feasibility, and risks

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-product-ideas.git
cd ai-product-ideas
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```bash
# For OpenAI
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7

# For Ollama (optional)
USE_OLLAMA=true
OLLAMA_MODEL=llama2  # or any other model you have pulled
OLLAMA_API_URL=http://localhost:11434/api
```

5. If using Ollama:
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Pull your desired model:
     ```bash
     ollama pull llama2  # or any other model
     ```

## Usage

1. Run the application:
```bash
python run.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter your product idea and any additional context

4. Click "Evaluate Idea" to get insights from multiple AI agents

## Available Agents

- **Sam Altman**: Focuses on market potential, business model, and growth strategy
- **Demis Hassabis**: Analyzes technical feasibility, AI/ML opportunities, and research potential

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
