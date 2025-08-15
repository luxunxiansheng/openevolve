"""
Prompt building utilities for LLM-based program evolution.
"""

from enum import Enum
from typing import List, Dict, Any

from opencontext.environment.templates import template_loader


class EvolutionMode(Enum):
    """
    Enum for evolution modes
    """
    FULL_REWRITE = "full_rewrite"
    DIFF = "diff"





class EvolutionPromptBuilder:
    """
    Simple prompt builder that combines generic and specific instructions

    This class builds system and user prompts for LLM-based program evolution,
    supporting different modes like full rewrite and diff-based evolution.
    """

    def __init__(self, language: str = "python"):
        """
        Initialize prompt builder

        Args:
                language: Programming language for code context
        """
        self.language = language

    def build_system_prompt(self,mode:EvolutionMode) -> str:
        """
        Build system prompt based on evolution mode

        Args:
                mode: Evolution mode (e.g., FULL_REWRITE, DIFF)

        Returns:
                System prompt string
        """
        if mode == EvolutionMode.FULL_REWRITE:
            return template_loader.load_template("full_rewrite")
        elif mode == EvolutionMode.DIFF:
            return template_loader.load_template("diff")
        else:
            raise ValueError(f"Unsupported evolution mode: {mode}")


    def build_user_prompt(
        self,
        current_program: str,
        elite_programs: List[Dict[str, Any]],
        current_metrics: Dict[str, float],
        mode: str = "full_rewrite",
    ) -> str:
        pass
   