"""
Simple PromptBuilder that uses existing templates and sampler for program evolution
"""

import logging
from typing import Any, Dict, List, Optional

from opencontext.prompt.sampler import PromptSampler
from opencontext.prompt.templates import Templates

logger = logging.getLogger(__name__)


class EvolutionPromptBuilder:
    """
    Simple prompt builder that uses existing PromptSampler and Templates
    for generating evolution-focused prompts.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize with existing PromptSampler infrastructure.

        Args:
            template_dir: Optional custom template directory
        """
        self.sampler = PromptSampler(template_dir=template_dir)

    def build_evolution_prompt(
        self,
        current_program: str,
        elite_programs: Optional[List[Dict[str, Any]]] = None,
        diverse_programs: Optional[List[Dict[str, Any]]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        language: str = "python",
        use_diff_format: bool = True,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Build an evolution prompt using existing templates.

        Args:
            current_program: The program to evolve
            elite_programs: Top-performing programs for context
            diverse_programs: Diverse programs for inspiration
            current_metrics: Current performance metrics
            language: Programming language
            use_diff_format: Whether to use diff format (SEARCH/REPLACE) or full rewrite
            **kwargs: Additional arguments for the sampler

        Returns:
            Dictionary with 'system' and 'user' messages
        """
        # Choose template based on evolution strategy
        if use_diff_format:
            user_template_key = Templates.DIFF_USER
        else:
            user_template_key = Templates.FULL_REWRITE_USER

        # Use ACTOR_SYSTEM for evolution tasks
        system_template_key = Templates.ACTOR_SYSTEM

        # Prepare program lists
        top_programs = elite_programs or []
        inspirations = diverse_programs or []

        # Build the prompt using existing sampler
        return self.sampler.build_prompt(
            user_template_key=user_template_key,
            current_program=current_program,
            program_metrics=current_metrics or {},
            top_programs=top_programs,
            inspirations=inspirations,
            language=language,
            system_template_key=system_template_key,
            **kwargs,
        )

    def build_diff_evolution_prompt(
        self,
        current_program: str,
        elite_programs: Optional[List[Dict[str, Any]]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        language: str = "python",
        **kwargs,
    ) -> Dict[str, str]:
        """
        Build a diff-based evolution prompt (SEARCH/REPLACE format).
        """
        return self.build_evolution_prompt(
            current_program=current_program,
            elite_programs=elite_programs,
            current_metrics=current_metrics,
            language=language,
            use_diff_format=True,
            **kwargs,
        )

    def build_rewrite_evolution_prompt(
        self,
        current_program: str,
        elite_programs: Optional[List[Dict[str, Any]]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        language: str = "python",
        **kwargs,
    ) -> Dict[str, str]:
        """
        Build a full rewrite evolution prompt.
        """
        return self.build_evolution_prompt(
            current_program=current_program,
            elite_programs=elite_programs,
            current_metrics=current_metrics,
            language=language,
            use_diff_format=False,
            **kwargs,
        )

    def build_evaluation_prompt(self, program: str, language: str = "python") -> Dict[str, str]:
        """
        Build an evaluation prompt using the CRITIC system and EVALUATION template.
        """
        return self.sampler.build_prompt(
            user_template_key=Templates.EVALUATION,
            current_program=program,
            language=language,
            system_template_key=Templates.CRITIC_SYSTEM,
        )
