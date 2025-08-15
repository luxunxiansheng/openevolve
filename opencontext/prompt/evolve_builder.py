"""
Simple PromptBuilder for OpenEvolve - Uses existing sampler and templates
"""

import logging
from typing import Any, Dict, List, Optional

from opencontext.prompt.sampler import PromptSampler
from opencontext.prompt.templates import Templates

logger = logging.getLogger(__name__)


class EvolvePromptBuilder:
    """
    Simple prompt builder that leverages existing PromptSampler and Templates
    for LLM-guided program evolution.
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        evolution_mode: str = "full_rewrite",  # "diff" or "full_rewrite"
        num_top_programs: int = 3,
        num_diverse_programs: int = 2,
        include_artifacts: bool = True,
    ):
        """
        Initialize using existing PromptSampler infrastructure.

        Args:
            template_dir: Directory containing custom templates
            evolution_mode: "diff" for incremental changes, "full_rewrite" for complete rewrites
            num_top_programs: Number of top-performing programs to include
            num_diverse_programs: Number of diverse programs for inspiration
            include_artifacts: Whether to include execution artifacts
        """
        self.evolution_mode = evolution_mode

        # Use ACTOR_SYSTEM for evolution-focused prompts
        system_template = Templates.ACTOR_SYSTEM

        # Initialize the existing PromptSampler
        self.sampler = PromptSampler(
            template_dir=template_dir,
            system_template_key=system_template,
            num_top_programs=num_top_programs,
            num_diverse_programs=num_diverse_programs,
            include_artifacts=include_artifacts,
        )

    def build_evolution_prompt(
        self,
        current_program: str,
        current_metrics: Dict[str, float],
        top_programs: Optional[List[Dict[str, Any]]] = None,
        diverse_programs: Optional[List[Dict[str, Any]]] = None,
        previous_programs: Optional[List[Dict[str, Any]]] = None,
        language: str = "python",
        problem_description: str = "",
        execution_artifacts: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Build an evolution prompt using existing templates and sampler.

        Args:
            current_program: Program to evolve
            current_metrics: Current performance metrics
            top_programs: List of high-performing programs
            diverse_programs: List of diverse programs for inspiration
            previous_programs: List of previous attempts
            language: Programming language
            problem_description: Optional problem description
            execution_artifacts: Execution feedback artifacts
            **kwargs: Additional template variables

        Returns:
            Dictionary with 'system' and 'user' prompt components
        """

        # Choose template based on evolution mode
        if self.evolution_mode == "diff":
            user_template_key = Templates.DIFF_USER
        else:
            user_template_key = Templates.FULL_REWRITE_USER

        # Add problem description to kwargs if provided
        if problem_description:
            kwargs["problem_description"] = problem_description

        # Use the existing sampler's build_prompt method
        return self.sampler.build_prompt(
            user_template_key=user_template_key,
            current_program=current_program,
            program_metrics=current_metrics,
            top_programs=top_programs or [],
            inspirations=diverse_programs or [],
            previous_programs=previous_programs or [],
            language=language,
            program_artifacts=execution_artifacts,
            **kwargs,
        )

    def build_critic_prompt(
        self, program_to_evaluate: str, language: str = "python", **kwargs
    ) -> Dict[str, str]:
        """
        Build a critic prompt for program evaluation using existing templates.

        Args:
            program_to_evaluate: Program to evaluate
            language: Programming language
            **kwargs: Additional template variables

        Returns:
            Dictionary with 'system' and 'user' prompt components
        """

        # Use CRITIC_SYSTEM template and EVALUATION user template
        critic_sampler = PromptSampler(
            system_template_key=Templates.CRITIC_SYSTEM,
            include_artifacts=False,
        )

        return critic_sampler.build_prompt(
            user_template_key=Templates.EVALUATION,
            current_program=program_to_evaluate,
            language=language,
            **kwargs,
        )

    def build_crossover_prompt(
        self, parent_a: Dict[str, Any], parent_b: Dict[str, Any], language: str = "python", **kwargs
    ) -> Dict[str, str]:
        """
        Build a crossover prompt by using the full_rewrite template
        with both parents as inspiration.

        Args:
            parent_a: First parent program
            parent_b: Second parent program
            language: Programming language
            **kwargs: Additional template variables

        Returns:
            Dictionary with 'system' and 'user' prompt components
        """

        # Use both parents as "top programs" for crossover inspiration
        combined_programs = [parent_a, parent_b]

        # Create a crossover instruction
        crossover_instruction = f"""
# Crossover Evolution Task

You are given two parent programs to combine into an improved offspring.
Analyze the strengths of each parent and intelligently combine their best features.

Parent A Performance: {parent_a.get('metrics', {})}
Parent B Performance: {parent_b.get('metrics', {})}

Create a new program that combines the best aspects of both parents.
"""

        kwargs["crossover_instruction"] = crossover_instruction

        return self.sampler.build_prompt(
            user_template_key=Templates.FULL_REWRITE_USER,
            current_program=parent_a.get("code", ""),  # Use first parent as base
            program_metrics=parent_a.get("metrics", {}),
            top_programs=combined_programs,
            language=language,
            **kwargs,
        )
