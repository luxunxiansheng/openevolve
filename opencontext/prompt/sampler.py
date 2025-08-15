"""
Prompt sampling for OpenEvolve
"""

import logging
import random
from typing import Any, Dict, List, Optional, Union

from opencontext.prompt.templates import TemplateManager, Templates
from opencontext.utils import safe_numeric_average

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution with configurable templates and formatting"""

    def __init__(
        self,
        template_dir: Optional[str] = None,
        system_template_key: str = Templates.BASE_SYSTEM,
        num_top_programs: int = 3,
        num_diverse_programs: int = 2,
        use_template_stochasticity: bool = True,
        template_variations: Optional[Dict[str, List[str]]] = None,
        use_meta_prompting: bool = False,
        meta_prompt_weight: float = 0.1,
        include_artifacts: bool = True,
        max_artifact_bytes: int = 20 * 1024,
        include_changes_under_chars: int = 100,
        concise_implementation_max_lines: int = 10,
        comprehensive_implementation_min_lines: int = 50,
        code_length_threshold: Optional[int] = 2048,
    ):
        """
        Initialize PromptSampler with configuration

        Args:
            template_dir: Directory containing templates
            system_template_key: Key for system message template
            num_top_programs: Number of top programs to include
            num_diverse_programs: Number of diverse programs to include
            use_template_stochasticity: Whether to apply template variations
            template_variations: Dictionary of template variations
            use_meta_prompting: Whether to use meta prompting
            meta_prompt_weight: Weight for meta prompting
            include_artifacts: Whether to include artifacts in prompts
            max_artifact_bytes: Maximum bytes for artifacts
            include_changes_under_chars: Include changes under this character limit
            concise_implementation_max_lines: Max lines for concise implementation
            comprehensive_implementation_min_lines: Min lines for comprehensive implementation
            code_length_threshold: Threshold for code length warnings
        """
        # Core settings
        self.template_dir = template_dir
        self.system_template_key = system_template_key

        # Configuration parameters
        self.num_top_programs = num_top_programs
        self.num_diverse_programs = num_diverse_programs
        self.use_template_stochasticity = use_template_stochasticity
        self.template_variations = template_variations or {}
        self.use_meta_prompting = use_meta_prompting
        self.meta_prompt_weight = meta_prompt_weight
        self.include_artifacts = include_artifacts
        self.max_artifact_bytes = max_artifact_bytes
        self.include_changes_under_chars = include_changes_under_chars
        self.concise_implementation_max_lines = concise_implementation_max_lines
        self.comprehensive_implementation_min_lines = comprehensive_implementation_min_lines
        self.code_length_threshold = code_length_threshold

        # Initialize components
        self.template_manager = TemplateManager(template_dir)

        # Setup
        random.seed()

    def build_prompt(
        self,
        user_template_key: str,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],
        language: str = "python",
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM

        Args:
            user_template_key: Template key for the user message
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs (best by fitness)
            inspirations: List of inspiration programs (diverse/creative examples)
            language: Programming language
            program_artifacts: Optional artifacts from program evaluation
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Validate and get templates
        user_template = self.template_manager.get_template(user_template_key)
        system_message = self.template_manager.get_template(self.system_template_key)

        # Fallback to minimal prompt if templates are missing
        if not user_template or not system_message:
            logger.warning(
                f"Missing templates: user='{user_template_key}', system='{self.system_template_key}'. Using fallback."
            )
            return {
                "system": "You are a helpful coding assistant.",
                "user": f"Please help improve this {language} code:\n\n{current_program}",
            }

        # Build prompt components with safe defaults
        components = self._build_prompt_components(
            current_program=current_program,
            parent_program=parent_program,
            program_metrics=program_metrics or {},
            previous_programs=previous_programs or [],
            top_programs=top_programs or [],
            inspirations=inspirations or [],
            language=language,
            program_artifacts=program_artifacts,
        )

        # Apply stochastic variations if enabled
        if self.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Format the final user message
        user_message = user_template.format(
            **components,
            **kwargs,
        )

        return {
            "system": system_message,
            "user": user_message,
        }

    def _build_prompt_components(
        self,
        current_program: str,
        parent_program: str,
        program_metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]],
    ) -> Dict[str, str]:
        """Build all prompt components"""

        metrics_str = self._format_metrics(program_metrics)
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language
        )

        artifacts_section = ""
        if self.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        return {
            "metrics": metrics_str,
            "improvement_areas": improvement_areas,
            "evolution_history": evolution_history,
            "current_program": current_program,
            "language": language,
            "artifacts": artifacts_section,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display"""
        return "\n".join([f"- {key}: {value}" for key, value in metrics.items()])

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
    ) -> str:
        """Identify potential areas for improvement"""
        improvement_areas = []

        # Compare with parent program if available
        if parent_program and parent_program != current_program:
            if len(current_program) > len(parent_program) * 1.5:
                improvement_areas.append(
                    "Current program is significantly larger than parent - consider simplification"
                )
            elif len(current_program) < len(parent_program) * 0.5:
                improvement_areas.append(
                    "Current program is much smaller than parent - may be missing functionality"
                )

        # Simple length check
        if self.code_length_threshold and len(current_program) > self.code_length_threshold:
            improvement_areas.append("Consider simplifying the code to improve readability")

        # Simple metric trend check
        if len(previous_programs) >= 2:
            recent_programs = previous_programs[-2:]
            for metric_name, current_value in metrics.items():
                if isinstance(current_value, (int, float)):
                    for program in recent_programs:
                        recent_value = program.get("metrics", {}).get(metric_name)
                        if isinstance(recent_value, (int, float)) and current_value < recent_value:
                            improvement_areas.append(f"Focus on improving {metric_name}")
                            break

        # Default if no areas found
        if not improvement_areas:
            improvement_areas.append("Focus on optimizing the code for better performance")

        return "\n".join([f"- {area}" for area in improvement_areas])

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        history_template = self.template_manager.get_template(Templates.EVOLUTION_HISTORY)
        previous_attempt_template = self.template_manager.get_template(Templates.PREVIOUS_ATTEMPT)
        top_program_template = self.template_manager.get_template(Templates.TOP_PROGRAM)

        # Format previous attempts
        previous_attempts_str = ""
        selected_previous = (
            previous_programs[-3:] if len(previous_programs) > 3 else previous_programs
        )

        for i, program in enumerate(reversed(selected_previous)):
            metrics = program.get("metrics", {})
            performance_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])

            # Generate more meaningful changes description
            program_code = program.get("code", "")
            if program_code:
                lines_count = len(program_code.split("\n"))
                changes = f"Program with {lines_count} lines of code"
            else:
                changes = "Program evolution attempt"

            # Determine outcome based on metrics
            if metrics:
                avg_score = sum(v for v in metrics.values() if isinstance(v, (int, float))) / len(
                    [v for v in metrics.values() if isinstance(v, (int, float))]
                )
                if avg_score >= 0.8:
                    outcome = "Strong performance"
                elif avg_score >= 0.6:
                    outcome = "Good performance"
                elif avg_score >= 0.4:
                    outcome = "Moderate performance"
                else:
                    outcome = "Needs improvement"
            else:
                outcome = "Performance unknown"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=len(previous_programs) - i,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: self.num_top_programs]

        for i, program in enumerate(selected_top):
            program_code = program.get("code", "")
            score = safe_numeric_average(program.get("metrics", {}))

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_code,
                    key_features="Performance optimized",
                )
                + "\n\n"
            )

        # Format inspirations
        inspirations_section_str = self._format_inspirations_section(inspirations, language)

        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=top_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self, inspirations: List[Dict[str, Any]], language: str
    ) -> str:
        """Format the inspirations section for the prompt"""
        if not inspirations:
            return ""

        inspirations_section_template = self.template_manager.get_template(
            Templates.INSPIRATIONS_SECTION
        )
        inspiration_program_template = self.template_manager.get_template(
            Templates.INSPIRATION_PROGRAM
        )

        inspiration_programs_str = ""

        for i, program in enumerate(inspirations):
            program_code = program.get("code", "")
            score = safe_numeric_average(program.get("metrics", {}))

            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type="Alternative",
                    language=language,
                    program_snippet=program_code,
                    unique_features="Different approach",
                )
                + "\n\n"
            )

        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template
        for key, variations in self.template_variations.items():
            if variations and f"{{{key}}}" in result:
                result = result.replace(f"{{{key}}}", random.choice(variations))
        return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """Render artifacts for prompt inclusion"""
        if not artifacts:
            return ""

        sections = []
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            if len(content) > self.max_artifact_bytes:
                content = content[: self.max_artifact_bytes] + "\n... (truncated)"
            sections.append(f"### {key}\n```\n{content}\n```")

        return "## Last Execution Output\n\n" + "\n\n".join(sections)

        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """Safely decode an artifact value to string"""
        if isinstance(value, str):
            return value
        elif isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        else:
            return str(value)
