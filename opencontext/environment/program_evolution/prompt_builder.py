"""
Prompt Builder Module

Builds prompts for program evolution with clear separation of context and instructions.
"""

from typing import Any, Dict, List, Tuple, Union

from opencontext.common.actions import EvolutionAction, EvolutionMode
from .template_manager import TemplateManager


class PromptBuilder:
    """Builds prompts for program evolution"""

    def __init__(
        self,
        language: str = "python",
        max_history_entries: int = 3,
        program_preview_length: int = 50,
        artifact_preview_length: int = 200,
    ):
        self.language = language
        self.max_history_entries = max_history_entries
        self.template_manager = TemplateManager(
            max_history_entries=max_history_entries,
            program_preview_length=program_preview_length,
            artifact_preview_length=artifact_preview_length,
        )

    def build_prompts(self, action: EvolutionAction) -> Tuple[str, str]:
        """Build system and user prompts with clear separation:
        - System prompt: Program context (current state, history, metrics)
        - User prompt: Clear instruction/task directive
        """
        # System prompt provides context about the program
        system_prompt = self._build_context_prompt(action)

        # User prompt is the clear instruction/directive
        user_prompt = self._build_instruction_prompt(action)

        return system_prompt, user_prompt

    def _build_context_prompt(self, action: EvolutionAction) -> str:
        """Build system prompt focusing on program context and state"""
        # Determine mode for template selection
        mode = EvolutionMode.DIFF if action.mode == "diff" else EvolutionMode.FULL_REWRITE
        template_name = mode.value  # "diff" or "full_rewrite"

        # Create context about the current program state
        context_info = {
            "current_program": action.current_program.code if action.current_program else "",
            "language": self.language,
            "metrics": self._format_metrics(
                action.current_program.metrics if action.current_program else {}
            ),
            "evolution_history": self._format_evolution_history_for_template(action),
            
            "artifacts": self._format_artifacts(
                action.current_program.metadata.get("artifacts", {})
                if action.current_program
                else {}
            ),
        }

        # Load template and inject context
        template = self.template_manager.load_template(template_name)

        # Replace placeholders with context information
        result = template
        for key, value in context_info.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def _build_instruction_prompt(self, action: EvolutionAction) -> str:
        """Build user prompt from goal and instructions"""
        prompt_parts = []

        # Add the main goal
        prompt_parts.append(f"Goal: {action.goal}")

        # Add specific instructions if provided
        if action.instructions:
            prompt_parts.append("Specific instructions:")
            for i, instruction in enumerate(action.instructions, 1):
                prompt_parts.append(f"{i}. {instruction}")
        else:
            prompt_parts.append("Please improve the current program.")

        # Add focus areas if provided
        if action.improvement_areas:
            prompt_parts.append(f"Improvement areas: {', '.join(action.improvement_areas)}")

        # Add constraints if provided
        if action.constraints:
            constraints_text = ", ".join([f"{k}: {v}" for k, v in action.constraints.items()])
            prompt_parts.append(f"Constraints: {constraints_text}")

        return "\n".join(prompt_parts)

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for template insertion"""
        if not metrics:
            return "No metrics available"

        formatted = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{name}: {value:.4f}")
            else:
                formatted.append(f"{name}: {value}")

        return ", ".join(formatted)

    def _format_artifacts(self, artifacts: Dict[str, Union[str, bytes, Any]]) -> str:
        """Format artifacts for template insertion"""
        if not artifacts:
            return ""

        artifact_parts = []
        for name, value in artifacts.items():
            if isinstance(value, str):
                # Text artifact
                preview = value[:200] + "..." if len(value) > 200 else value
                artifact_parts.append(f"**{name}**: {preview}")
            elif isinstance(value, bytes):
                # Binary artifact
                artifact_parts.append(f"**{name}**: Binary data ({len(value)} bytes)")
            else:
                # Other types
                artifact_parts.append(f"**{name}**: {type(value).__name__} data")

        return "\n".join(artifact_parts)

    def _format_evolution_history_for_template(self, action: EvolutionAction) -> str:
        """Format evolution history for template insertion"""
        if not action.previous_programs:
            return "No previous evolution history available."

        history_parts = []
        for i, program in enumerate(action.previous_programs[-self.max_history_entries :], 1):
            metrics = program.metrics or {}
            program_preview = program.code

            # Use explicit outcome if provided, otherwise just show metrics
            outcome = program.metadata.get("outcome", "See metrics")

            # Create program preview
            if isinstance(program_preview, str) and len(program_preview) > 50:
                program_preview = program_preview[:50] + "..."

            # Format metrics concisely
            metrics_str = (
                ", ".join(
                    [
                        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in metrics.items()
                    ]
                )
                if metrics
                else "No metrics"
            )

            history_parts.append(
                f"**Attempt {i}:**\n- Performance: {metrics_str}\n- Outcome: {outcome}\n- Preview: {program_preview}"
            )

        return "\n\n".join(history_parts)

    def _format_evolution_history(self, action: EvolutionAction) -> List[Dict[str, Any]]:
        """Format evolution history from action data"""
        # Use previous_programs from the new EvolutionAction structure
        previous_programs = action.previous_programs or []

        if not previous_programs:
            return []

        # Return the most recent programs as structured data
        history_entries = []
        for i, program in enumerate(previous_programs[-self.max_history_entries :], 1):
            history_entries.append(
                {
                    "attempt": i,
                    "program": program.code,
                    "metrics": program.metrics,
                    "outcome": program.metadata.get("outcome", "See metrics"),
                }
            )

        return history_entries
