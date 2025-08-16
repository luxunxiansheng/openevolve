"""
Common action classes for OpenContext

This module contains shared action data structures used across different components
like environments, agents, and orchestrators.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from .program import Program


class EvolutionMode(Enum):
    """Evolution modes for code generation"""

    FULL_REWRITE = "full_rewrite"
    DIFF = "diff"


@dataclass
class EvolutionAction:
    """
    Action for program evolution with clear separation of context and instruction

    The program fields provide CONTEXT - what the LLM should understand
    The goal and task fields provide INSTRUCTION - what the LLM should achieve and how
    """

    # INSTRUCTION: What the LLM should achieve and how
    goal: str  # High-level objective like "improve performance", "enhance readability"
    instructions: Optional[List[str]] = (
        None  # Specific actions like ["replace recursion with iteration", "add docstrings", "add type hints"]
    )

    # CONTEXT: What the LLM should understand
    current_program: Optional[Program] = None  # The program to evolve
    parent_program: Optional[Program] = None  # Original/parent for reference
    previous_programs: Optional[List[Program]] = None  # Evolution history

    # EVOLUTION SETTINGS
    mode: EvolutionMode = EvolutionMode.FULL_REWRITE  # How to evolve (diff/rewrite)

    # ADDITIONAL CONTEXT
    constraints: Optional[Dict[str, Any]] = None  # Any constraints or requirements
    improvement_areas: Optional[List[str]] = None  # Specific areas to improve

    def __post_init__(self):
        """Initialize default values"""
        if self.previous_programs is None:
            self.previous_programs = []
        if self.constraints is None:
            self.constraints = {}
        if self.improvement_areas is None:
            self.improvement_areas = []
        if self.instructions is None:
            self.instructions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation with proper nested serialization"""
        return asdict(self)

    def get_context_summary(self) -> str:
        """Get a summary of the context for logging/debugging"""
        context_parts = []
        if self.current_program:
            context_parts.append(f"Current: {self.current_program.id}")
        if self.parent_program:
            context_parts.append(f"Parent: {self.parent_program.id}")
        if self.previous_programs:
            context_parts.append(f"History: {len(self.previous_programs)} programs")
        if self.improvement_areas:
            context_parts.append(f"Improvement areas: {', '.join(self.improvement_areas)}")

        return " | ".join(context_parts) if context_parts else "No context"

    def get_instruction_summary(self) -> str:
        """Get a summary of the instruction for logging/debugging"""
        goal_part = f"Goal: {self.goal[:30]}{'...' if len(self.goal) > 30 else ''}"

        if self.instructions:
            instructions_text = "; ".join(self.instructions)
            instruction_part = f"Instructions: {instructions_text[:50]}{'...' if len(instructions_text) > 50 else ''}"
        else:
            instruction_part = "Instructions: None"

        return f"{goal_part} | {instruction_part}"
