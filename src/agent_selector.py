from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel
import json
import os
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import print as rprint


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str
    description: str
    enabled: bool = True
    priority: int = 0  # Higher priority agents are evaluated first


class AgentSelector:
    """Manages the selection of agents for product evaluation."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the agent selector.
        
        Args:
            config_file: Optional path to a JSON config file to load/save agent settings
        """
        self.config_file = config_file or "agent_config.json"
        self.console = Console()
        
        # Default agent configurations
        self.agents = {
            "sam_altman": AgentConfig(
                name="Sam Altman",
                description="Evaluates startup potential, market fit, and technical feasibility",
                priority=1
            ),
            "demis_hassabis": AgentConfig(
                name="Demis Hassabis",
                description="Analyzes scientific breakthroughs, technical advancement, and research feasibility",
                priority=2
            ),
            "elon_musk": AgentConfig(
                name="Elon Musk",
                description="Assesses innovation potential, market disruption, and technical feasibility",
                priority=3
            ),
            "adam_dangelo": AgentConfig(
                name="Adam D'Angelo",
                description="Evaluates platform potential, AI infrastructure, and social impact",
                priority=4
            ),
            "daniel_gross": AgentConfig(
                name="Daniel Gross",
                description="Analyzes startup potential, AI infrastructure, and market fit",
                priority=5
            ),
            "sebastian_thrun": AgentConfig(
                name="Sebastian Thrun",
                description="Evaluates autonomous systems, educational impact, and innovation potential",
                priority=6
            ),
            "emad_mostaque": AgentConfig(
                name="Emad Mostaque",
                description="Assesses AI infrastructure, community impact, and open source potential",
                priority=7
            ),
            "clement_delangue": AgentConfig(
                name="Clement Delangue",
                description="Analyzes AI innovation, technical feasibility, and practical applications",
                priority=8
            ),
            "project_advisor": AgentConfig(
                name="Project Advisor",
                description="Provides final recommendation and timeline based on all other agents",
                priority=9,
                enabled=True  # Always enabled by default
            )
        }
        
        # Load configuration if file exists
        self._load_config()
    
    def _load_config(self) -> None:
        """Load agent configuration from file if it exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    for agent_id, config in config_data.items():
                        if agent_id in self.agents:
                            self.agents[agent_id].enabled = config.get("enabled", True)
                            self.agents[agent_id].priority = config.get("priority", 0)
            except Exception as e:
                self.console.print(f"[red]Error loading config: {e}[/red]")
    
    def _save_config(self) -> None:
        """Save current agent configuration to file."""
        try:
            config_data = {
                agent_id: {
                    "enabled": agent.enabled,
                    "priority": agent.priority
                }
                for agent_id, agent in self.agents.items()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            self.console.print(f"[red]Error saving config: {e}[/red]")
    
    def display_agent_table(self) -> None:
        """Display a table of all available agents and their status."""
        table = Table(title="Available Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")
        table.add_column("Enabled", style="yellow")
        table.add_column("Priority", style="magenta")
        
        # Sort agents by priority
        sorted_agents = sorted(
            self.agents.items(), 
            key=lambda x: (x[1].priority, x[0])
        )
        
        for agent_id, agent in sorted_agents:
            table.add_row(
                agent_id,
                agent.name,
                agent.description,
                "✓" if agent.enabled else "✗",
                str(agent.priority)
            )
        
        self.console.print(table)
    
    def toggle_agent(self, agent_id: str) -> bool:
        """
        Toggle the enabled status of an agent.
        
        Args:
            agent_id: The ID of the agent to toggle
            
        Returns:
            bool: True if the agent was toggled, False if the agent doesn't exist
        """
        if agent_id in self.agents:
            # Don't allow toggling the project advisor
            if agent_id == "project_advisor":
                self.console.print("[yellow]The Project Advisor cannot be disabled as it's required for final recommendations.[/yellow]")
                return False
                
            self.agents[agent_id].enabled = not self.agents[agent_id].enabled
            self._save_config()
            return True
        return False
    
    def set_agent_priority(self, agent_id: str, priority: int) -> bool:
        """
        Set the priority of an agent.
        
        Args:
            agent_id: The ID of the agent
            priority: The new priority value (higher numbers = higher priority)
            
        Returns:
            bool: True if the priority was set, False if the agent doesn't exist
        """
        if agent_id in self.agents:
            self.agents[agent_id].priority = priority
            self._save_config()
            return True
        return False
    
    def get_enabled_agents(self) -> Dict[str, AgentConfig]:
        """Get a dictionary of all enabled agents."""
        return {
            agent_id: agent 
            for agent_id, agent in self.agents.items() 
            if agent.enabled
        }
    
    def interactive_selection(self) -> None:
        """Run an interactive selection process for agents."""
        self.console.print("[bold green]Agent Selection[/bold green]")
        self.console.print("Select which agents to include in your product evaluation.")
        
        while True:
            self.display_agent_table()
            
            self.console.print("\n[bold]Options:[/bold]")
            self.console.print("1. Toggle agent")
            self.console.print("2. Set agent priority")
            self.console.print("3. Save and exit")
            
            choice = Prompt.ask("Enter your choice", choices=["1", "2", "3"])
            
            if choice == "1":
                agent_id = Prompt.ask("Enter agent ID to toggle")
                if self.toggle_agent(agent_id):
                    self.console.print(f"[green]Agent {agent_id} toggled successfully.[/green]")
                else:
                    self.console.print(f"[red]Agent {agent_id} not found.[/red]")
            
            elif choice == "2":
                agent_id = Prompt.ask("Enter agent ID")
                try:
                    priority = int(Prompt.ask("Enter new priority (higher number = higher priority)"))
                    if self.set_agent_priority(agent_id, priority):
                        self.console.print(f"[green]Priority for {agent_id} set to {priority}.[/green]")
                    else:
                        self.console.print(f"[red]Agent {agent_id} not found.[/red]")
                except ValueError:
                    self.console.print("[red]Priority must be a number.[/red]")
            
            elif choice == "3":
                self._save_config()
                self.console.print("[green]Configuration saved. Exiting...[/green]")
                break
    
    def get_agent_functions(self, agent_functions: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Get the agent functions for enabled agents.
        
        Args:
            agent_functions: Dictionary mapping agent IDs to their functions
            
        Returns:
            Dict[str, Callable]: Dictionary of enabled agent functions
        """
        enabled_agents = self.get_enabled_agents()
        return {
            agent_id: agent_functions[agent_id]
            for agent_id in enabled_agents
            if agent_id in agent_functions
        } 