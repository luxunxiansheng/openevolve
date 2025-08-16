"""
Program Evolution Engine Module

Main orchestrator for LLM-based program evolution.
"""


from opencontext.common.actions import EvolutionAction
from opencontext.llm.llm_interface import LLMInterface
from .prompt_builder import PromptBuilder
from .program_extractor import ProgramExtractor


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

    async def generate_code(self, action: EvolutionAction) -> str:
        """Generate improved code from evolution action"""
        system_prompt, user_prompt = self.prompt_builder.build_prompts(action)
        response = await self.llm.generate(user_prompt, system_message=system_prompt)
        return self._extract_code_from_response(response, action.mode.value)

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
