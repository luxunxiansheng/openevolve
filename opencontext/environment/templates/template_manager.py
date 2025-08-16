"""
Template Manager for OpenContext Environment

Manages loading and caching of template files for various components
like program evolution, evaluation, etc.
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
            templates_dir = Path(__file__).parent  # templates folder itself
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
