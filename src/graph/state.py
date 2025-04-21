from typing import Dict, Any, Optional
import json

class AgentState:
    """Simple state container for agent data."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.reasoning = []
    
    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
    
    def add_reasoning(self, agent: str, reasoning: str) -> None:
        """Add reasoning from an agent to the state."""
        self.reasoning.append({
            "agent": agent,
            "reasoning": reasoning
        })
    
    def get_reasoning(self) -> list:
        """Get all reasoning from agents."""
        return self.reasoning
    
    def to_json(self) -> str:
        """Convert state to JSON string."""
        return json.dumps({
            "data": self.data,
            "reasoning": self.reasoning
        }, indent=2)


def show_agent_reasoning(state: AgentState, agent: str) -> None:
    """Helper function to display agent reasoning."""
    for item in state.get_reasoning():
        if item["agent"] == agent:
            print(f"\n{agent} reasoning:")
            print(item["reasoning"])
            print("-" * 50)
