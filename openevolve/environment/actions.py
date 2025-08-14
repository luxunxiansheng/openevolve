"""
Evolution action structures for program evolution environment.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class EvolutionAction:
    """Structured action for program evolution

    Represents a single evolution instruction with context about the current state,
    performance metrics, and evolution mode preferences.
    """

    instruction: str
    current_program: Optional[str] = None
    current_score: Optional[float] = None
    parent_program: Optional[str] = None
    previous_attempts: Optional[list] = None
    context: Optional[Dict[str, Any]] = None
    mode: str = "full_rewrite"  # "diff" or "full_rewrite"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionAction":
        """Create action from dictionary"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary"""
        return {
            "instruction": self.instruction,
            "current_program": self.current_program,
            "current_score": self.current_score,
            "parent_program": self.parent_program,
            "previous_attempts": self.previous_attempts,
            "context": self.context,
            "mode": self.mode,
        }

    def validate(self) -> bool:
        """Validate that the action has required fields"""
        return bool(self.instruction and self.instruction.strip())

    def get_context_summary(self) -> str:
        """Get a summary of the context for logging/debugging"""
        parts = []
        if self.current_score is not None:
            parts.append(f"score={self.current_score:.4f}")
        if self.context:
            parts.append(f"context_keys={list(self.context.keys())}")
        if self.previous_attempts:
            parts.append(f"attempts={len(self.previous_attempts)}")
        return f"EvolutionAction({', '.join(parts)})"
