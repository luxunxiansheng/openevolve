import asyncio
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from opencontext.common.actions import EvolutionAction
from opencontext.llm.llm_interface import LLMInterface


class EvolutionMode(Enum):
    """Evolution modes for code generation"""

    FULL_REWRITE = "full_rewrite"
    DIFF = "diff"


@dataclass
class PromptContext:
    """Context data for prompt generation"""

    current_program: str
    metrics: Optional[Dict[str, float]] = None
    evolution_history: Optional[List[Dict[str, Any]]] = None
    improvement_areas: str = ""
    artifacts: Optional[Dict[str, Union[str, bytes, Any]]] = None
    language: str = "python"

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.artifacts is None:
            self.artifacts = {}
        if self.evolution_history is None:
            self.evolution_history = []


@dataclass
class ExtractedProgram:
    """Result of program extraction from LLM response"""

    success: bool
    program: str = ""
    changes: Optional[List[Dict[str, str]]] = None
    error: str = ""

    def __post_init__(self):
        if self.changes is None:
            self.changes = []


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
            templates_dir = Path(__file__).parent / "templates"
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

    def format_template(self, template_name: str, context: PromptContext, **kwargs) -> str:
        """Load and format a template with context"""
        template = self.load_template(template_name)

        # Create safe formatting dict
        format_dict = {
            "current_program": context.current_program,
            "metrics": self._format_metrics(context.metrics or {}),
            "evolution_history": self._format_evolution_history(context.evolution_history or []),
            "improvement_areas": context.improvement_areas,
            "artifacts": self._format_artifacts(context.artifacts or {}),
            "language": context.language,
            **kwargs,
        }

        # Replace placeholders
        result = template
        for key, value in format_dict.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

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


class ProgramExtractor:
    """Extracts generated programs from LLM responses"""

    @staticmethod
    def extract_full_rewrite(response: str, language: str = "python") -> ExtractedProgram:
        """Extract program from full rewrite response"""
        # Try to find code blocks
        code_block_pattern = rf"```{language}\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return ExtractedProgram(success=True, program=matches[0].strip())

        # Fallback to any code block
        code_block_pattern = r"```(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return ExtractedProgram(success=True, program=matches[0].strip())

        # If no code blocks, return the response as-is
        return ExtractedProgram(success=True, program=response.strip())

    @staticmethod
    def extract_diff_changes(response: str) -> ExtractedProgram:
        """Extract changes from diff-based response"""
        # Try JSON first (preferred format)
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                changes = []
                for item in parsed:
                    if isinstance(item, dict) and "search" in item and "replace" in item:
                        changes.append(
                            {
                                "search": item["search"],
                                "replace": item["replace"],
                                "rationale": item.get("rationale", ""),
                            }
                        )
                return ExtractedProgram(success=True, changes=changes)
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback to SEARCH/REPLACE format
        diff_pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        matches = re.findall(diff_pattern, response, re.DOTALL)

        if matches:
            changes = []
            for search, replace in matches:
                changes.append(
                    {
                        "search": search.strip(),
                        "replace": replace.strip(),
                        "rationale": "Extracted from SEARCH/REPLACE format",
                    }
                )
            return ExtractedProgram(success=True, changes=changes)

        return ExtractedProgram(success=False, error="No valid changes found")

    @staticmethod
    def apply_diff_changes(original_program: str, changes: List[Dict[str, str]]) -> str:
        """Apply diff changes to original program"""
        result = original_program

        for change in changes:
            search_text = change.get("search", "")
            replace_text = change.get("replace", "")

            if search_text and search_text in result:
                result = result.replace(search_text, replace_text, 1)

        return result


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
        """Build system and user prompts from evolution action"""
        # Determine mode
        mode = EvolutionMode.DIFF if action.mode == "diff" else EvolutionMode.FULL_REWRITE

        # Create context from action
        context = PromptContext(
            current_program=action.current_program or "",
            metrics=(
                getattr(action, "current_metrics", None) or action.context.get("metrics", {})
                if action.context
                else {}
            ),
            evolution_history=self._format_evolution_history(action),
            improvement_areas=(
                getattr(action, "improvement_areas", None)
                or action.context.get("improvement_areas", "")
                if action.context
                else ""
            ),
            artifacts=(
                action.context.get("artifacts", {}) if action.context else {}
            ),  # Get from context or empty dict
            language=self.language,
        )

        # Build system prompt (template acts as system message)
        system_prompt = self._build_system_prompt(mode, context)

        # Build user prompt (specific instruction from action)
        user_prompt = action.instruction or "Please improve the current program."

        return system_prompt, user_prompt

    def _build_system_prompt(self, mode: EvolutionMode, context: PromptContext) -> str:
        """Build system prompt using template"""
        template_name = mode.value  # "diff" or "full_rewrite"
        return self.template_manager.format_template(template_name, context)

    def _format_evolution_history(self, action: EvolutionAction) -> List[Dict[str, Any]]:
        """Format evolution history from action data"""
        # Try to get previous programs from context or previous_attempts
        previous_programs = []
        if action.context and "previous_programs" in action.context:
            previous_programs = action.context["previous_programs"]
        elif action.previous_attempts:
            # Convert previous_attempts to program format
            previous_programs = []
            for i, attempt in enumerate(action.previous_attempts):
                if isinstance(attempt, dict):
                    previous_programs.append(attempt)
                else:
                    previous_programs.append({"program": str(attempt), "metrics": {}})

        if not previous_programs:
            return []

        # Return the most recent programs as structured data
        history_entries = []
        for i, program_data in enumerate(previous_programs[-self.max_history_entries :], 1):
            metrics = program_data.get("metrics", {})

            history_entries.append(
                {
                    "attempt": i,
                    "program": program_data.get("program", program_data.get("code", "")),
                    "metrics": metrics,
                    "outcome": program_data.get("outcome", "See metrics"),
                }
            )

        return history_entries


class ProgramEvolutionEngine:
    """Complete engine for LLM-based program evolution"""

    def __init__(
        self,
        llm: LLMInterface,
        language: str = "python",
        max_code_length: int = 20480,
        max_history_entries: int = 3,
        program_preview_length: int = 50,
        artifact_preview_length: int = 200,
    ):
        self.llm = llm
        self.language = language
        self.max_code_length = max_code_length
        self.prompt_builder = PromptBuilder(
            language=language,
            max_history_entries=max_history_entries,
            program_preview_length=program_preview_length,
            artifact_preview_length=artifact_preview_length,
        )
        self.program_extractor = ProgramExtractor()

    def generate_code(self, action: EvolutionAction) -> str:
        """Generate improved code from evolution action"""
        # Build prompts
        system_prompt, user_prompt = self.prompt_builder.build_prompts(action)

        # Run async generation in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self.llm.generate(user_prompt, system_message=system_prompt)
            )
            # Delegate extraction based on requested mode to keep behavior consistent
            return self._extract_code_from_response(response, action.mode)
        finally:
            loop.close()

    async def generate_code_async(self, action: EvolutionAction) -> str:
        """Generate improved code asynchronously"""
        system_prompt, user_prompt = self.prompt_builder.build_prompts(action)
        response = await self.llm.generate(user_prompt, system_message=system_prompt)
        return self._extract_code_from_response(response, action.mode)

    def _extract_code_from_response(self, response: str, mode: str) -> str:
        """Extract final code from LLM response"""
        if mode == "diff":
            # Extract changes and apply them
            extracted = self.program_extractor.extract_diff_changes(response)
            if extracted.success and extracted.changes:
                # For now, return the changes as a formatted string
                # In practice, you'd apply them to the original code
                changes_text = "\n".join(
                    [
                        f"Change: {change['search']} -> {change['replace']}"
                        for change in extracted.changes
                    ]
                )
                return changes_text
            else:
                raise ValueError(f"Failed to extract diff changes: {extracted.error}")
        else:
            # Full rewrite
            extracted = self.program_extractor.extract_full_rewrite(response, self.language)
            if extracted.success:
                new_code = extracted.program or response
                # Ensure evolved code is enclosed in EVOLVE block comments
                return self._enclose_code_block(new_code)
            else:
                raise ValueError("Failed to extract code from full rewrite response")

    def _enclose_code_block(self, code: str) -> str:
        """Wrap code with EVOLVE-BLOCK markers if they are not already present.

        Uses language-appropriate single-line comment prefix (defaults to '#').
        Strips surrounding triple-backtick fences if present before wrapping.
        """
        if "EVOLVE-BLOCK-START" in code and "EVOLVE-BLOCK-END" in code:
            return code

        lang = (self.language or "").lower()
        comment_prefix = "#" if lang in ("python", "py", "bash", "sh") else "//"

        # Strip common code fence wrappers
        stripped = code.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # remove first fence line
            inner = (
                "\n".join(lines[1:-1])
                if len(lines) > 2 and lines[-1].strip().startswith("```")
                else "\n".join(lines[1:])
            )
        else:
            inner = stripped

        start = f"{comment_prefix} EVOLVE-BLOCK-START"
        end = f"{comment_prefix} EVOLVE-BLOCK-END"
        return f"{start}\n{inner}\n{end}"
