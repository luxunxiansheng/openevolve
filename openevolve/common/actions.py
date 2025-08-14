"""
Common action classes for OpenEvolve

This module contains shared action data structures used across different components
like environments, agents, and orchestrators.
"""

from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, asdict


@dataclass
class EvolutionAction:
    """
    Structured action for program evolution

    This class represents an evolution instruction that can be used by both
    environments and agents to communicate evolution requests.

    Attributes:
        instruction: The specific evolution instruction/task
        current_program: The current program code to evolve
        current_score: Performance score of the current program
        parent_program: Parent program for reference/comparison
        previous_attempts: List of previous evolution attempts
        context: Additional context like focus areas, constraints, etc.
        mode: Evolution mode - "diff" for targeted changes, "full_rewrite" for complete rewrite
    """

    instruction: str
    current_program: Optional[str] = None
    current_score: Optional[float] = None
    parent_program: Optional[str] = None
    previous_attempts: Optional[List[Union[str, Dict[str, Any]]]] = None
    context: Optional[Dict[str, Any]] = None
    mode: str = "full_rewrite"  # "diff" or "full_rewrite"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionAction":
        """Create action from dictionary"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary"""
        return asdict(self)

    def validate(self) -> None:
        """Validate action parameters"""
        if not self.instruction or not self.instruction.strip():
            raise ValueError("Instruction cannot be empty")

        if self.mode not in ["diff", "full_rewrite"]:
            raise ValueError(f"Mode must be 'diff' or 'full_rewrite', got: {self.mode}")

        if self.current_score is not None and not isinstance(self.current_score, (int, float)):
            raise ValueError("Current score must be a number")

    def with_context(self, **kwargs) -> "EvolutionAction":
        """Create a new action with additional context"""
        new_context = self.context.copy() if self.context else {}
        new_context.update(kwargs)

        return EvolutionAction(
            instruction=self.instruction,
            current_program=self.current_program,
            current_score=self.current_score,
            parent_program=self.parent_program,
            previous_attempts=self.previous_attempts,
            context=new_context,
            mode=self.mode,
        )

    def __str__(self) -> str:
        """String representation for debugging"""
        parts = [f"EvolutionAction(mode={self.mode})"]
        parts.append(f"  instruction: {self.instruction[:50]}...")
        if self.current_score is not None:
            parts.append(f"  current_score: {self.current_score}")
        if self.context:
            parts.append(f"  context_keys: {list(self.context.keys())}")
        return "\n".join(parts)
