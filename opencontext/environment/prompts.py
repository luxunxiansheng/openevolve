"""
Prompt building utilities for LLM-based program evolution.
"""

from typing import Tuple, List, Dict, Any, Union
from opencontext.common.actions import EvolutionAction


class PromptBuilder:
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

    def build_system_prompt(self, mode: str = "full_rewrite") -> str:
        """
        Build generic system prompt for evolution

        Args:
                mode: Evolution mode - "diff" or "full_rewrite"

        Returns:
                System prompt string
        """
        if mode == "diff":
            return self._build_diff_system_prompt()
        else:
            return self._build_rewrite_system_prompt()

    def _build_diff_system_prompt(self) -> str:
        """Build system prompt for diff mode"""
        return """You are an expert software developer focused on iterative code improvement.
Your task is to analyze the current program and suggest targeted improvements using SEARCH/REPLACE diffs.

Use this exact format for changes:
<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Make targeted improvements that will enhance performance, readability, or efficiency."""

    def _build_rewrite_system_prompt(self) -> str:
        """Build system prompt for full rewrite mode"""
        return """You are an expert software developer focused on iterative code improvement.
Your task is to analyze the current program and rewrite it to improve performance, readability, or efficiency.

Provide the complete improved program code. Maintain the same functionality but with better implementation."""

    def build_user_prompt(self, action: EvolutionAction) -> str:
        """
        Build user prompt combining action-specific and generic instructions

        Args:
                action: Evolution action containing instruction and context

        Returns:
                User prompt string
        """
        parts = []

        # Add main task instruction
        parts.append(self._build_task_section(action.instruction))

        # Add current program context
        if action.current_program:
            parts.append(self._build_program_section(action.current_program))

        # Add performance context
        if action.current_score is not None:
            parts.append(self._build_performance_section(action.current_score))

        # Add parent program context
        if action.parent_program:
            parts.append(self._build_parent_section(action.parent_program))

        # Add previous attempts context
        if action.previous_attempts:
            parts.append(self._build_attempts_section(action.previous_attempts))

        # Add additional context
        if action.context:
            parts.append(self._build_context_section(action.context))

        # Add mode-specific instructions
        parts.append(self._build_instructions_section(action.mode))

        return "\n".join(parts)

    def _build_task_section(self, instruction: str) -> str:
        """Build task section of prompt"""
        return f"# Task\n{instruction}"

    def _build_program_section(self, program: str) -> str:
        """Build current program section"""
        return f"\n# Current Program\n```{self.language}\n{program}\n```"

    def _build_performance_section(self, score: float) -> str:
        """Build performance section"""
        return f"\n# Current Performance\n- Score: {score:.4f}"

    def _build_parent_section(self, parent_program: str) -> str:
        """Build parent program section"""
        return f"\n# Parent Program (for reference)\n```{self.language}\n{parent_program}\n```"

    def _build_attempts_section(self, attempts: List[Union[str, Dict[str, Any]]]) -> str:
        """Build previous attempts section"""
        parts = ["\n# Previous Attempts"]

        # Show last 3 attempts to avoid overwhelming the prompt
        recent_attempts = attempts[-3:] if len(attempts) > 3 else attempts

        for i, attempt in enumerate(recent_attempts):
            if isinstance(attempt, dict):
                description = attempt.get("description", "Unknown attempt")
                parts.append(f"- Attempt {i+1}: {description}")
            else:
                parts.append(f"- Attempt {i+1}: {attempt}")

        return "\n".join(parts)

    def _build_context_section(self, context: Dict[str, Any]) -> str:
        """Build additional context section"""
        context_items = []

        for key, value in context.items():
            if key == "focus_areas" and isinstance(value, list):
                context_items.append(f"Focus Areas: {', '.join(value)}")
            elif key == "constraints":
                context_items.append(f"Constraints: {value}")
            elif key == "artifacts":
                # Skip artifacts for now - could be handled separately
                continue
            else:
                context_items.append(f"{key}: {value}")

        if context_items:
            return f"\n# Additional Context\n" + "\n".join([f"- {item}" for item in context_items])

        return ""

    def _build_instructions_section(self, mode: str) -> str:
        """Build mode-specific instructions section"""
        if mode == "diff":
            return (
                f"\n# Instructions\n"
                f"Provide targeted improvements using SEARCH/REPLACE diff format. "
                f"Focus on specific enhancements to the existing code."
            )
        else:
            return (
                f"\n# Instructions\n"
                f"Provide a complete rewritten version of the program with improvements. "
                f"Ensure the same functionality but better implementation."
            )

    def build_prompts(self, action: EvolutionAction) -> tuple[str, str]:
        """
        Build both system and user prompts

        Args:
                action: Evolution action containing instruction and context

        Returns:
                Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.build_system_prompt(action.mode)
        user_prompt = self.build_user_prompt(action)
        return system_prompt, user_prompt
