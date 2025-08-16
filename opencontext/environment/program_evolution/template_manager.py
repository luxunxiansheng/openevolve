"""
Template Manager Module

Manages prompt templates for program evolution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TemplateManager:
    """Manages prompt templates for evolution"""

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        max_history_entries: int = 3,
        program_preview_length: int = 50,
        artifact_preview_length: int = 200,
    ):
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates"
        self.templates_dir = templates_dir
        self.max_history_entries = max_history_entries
        self.program_preview_length = program_preview_length
        self.artifact_preview_length = artifact_preview_length
        self._templates_cache = {}

    def load_template(self, template_name: str) -> str:
        """Load a template by name"""
        if template_name in self._templates_cache:
            return self._templates_cache[template_name]

        template_path = self.templates_dir / f"{template_name}.md"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        content = template_path.read_text()
        self._templates_cache[template_name] = content
        return content

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
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

    def _format_evolution_history(self, evolution_history: List[Dict[str, Any]]) -> str:
        """Format evolution history for template insertion"""
        if not evolution_history:
            return "No previous evolution history available."

        history_parts = []
        for i, program_data in enumerate(evolution_history[-self.max_history_entries :], 1):
            metrics = program_data.get("metrics", {})
            program_preview = program_data.get("program", program_data.get("code", "N/A"))

            # Use explicit outcome if provided, otherwise just show metrics
            outcome = program_data.get("outcome", "See metrics")

            # Create program preview
            if (
                isinstance(program_preview, str)
                and len(program_preview) > self.program_preview_length
            ):
                program_preview = program_preview[: self.program_preview_length] + "..."

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

    def _format_artifacts(self, artifacts: Dict[str, Union[str, bytes, Any]]) -> str:
        """Format artifacts for template insertion"""
        if not artifacts:
            return ""

        artifact_parts = []
        for name, value in artifacts.items():
            if isinstance(value, str):
                # Text artifact
                preview = (
                    value[: self.artifact_preview_length] + "..."
                    if len(value) > self.artifact_preview_length
                    else value
                )
                artifact_parts.append(f"**{name}**: {preview}")
            elif isinstance(value, bytes):
                # Binary artifact
                artifact_parts.append(f"**{name}**: Binary data ({len(value)} bytes)")
            else:
                # Other types
                artifact_parts.append(f"**{name}**: {type(value).__name__} data")

        return "\n".join(artifact_parts)
