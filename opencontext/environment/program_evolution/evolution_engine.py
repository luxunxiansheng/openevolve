"""
Program Evolution Engine Module

Main orchestrator for LLM-based program evolution.
"""

import logging
import time
from typing import Optional

from opencontext.common.actions import EvolutionAction, EvolutionMode
from opencontext.llm.llm_interface import LLMInterface
from .prompt_builder import PromptBuilder
from .program_extractor import ProgramExtractor


class ProgramEvolutionEngine:
    

    def __init__(
        self,
        llm: LLMInterface,
        language: str = "python",
        max_code_length: int = 20480,
        max_history_entries: int = 3,
        program_preview_length: int = 50,
        artifact_preview_length: int = 200,
        logger: Optional[logging.Logger] = None,
    ):
        self.llm = llm
        self.language = language
        self.max_code_length = max_code_length
        self.logger = logger or logging.getLogger(__name__)

        self.prompt_builder = PromptBuilder(
            language=language,
            max_history_entries=max_history_entries,
            program_preview_length=program_preview_length,
            artifact_preview_length=artifact_preview_length,
        )
        self.program_extractor = ProgramExtractor()

        self.logger.info(
            "ProgramEvolutionEngine initialized",
            extra={
                "language": language,
                "max_code_length": max_code_length,
                "max_history_entries": max_history_entries,
            },
        )

    async def generate_code(self, action: EvolutionAction) -> str:
        """Generate improved code from evolution action"""
        start_time = time.time()

        self.logger.info(
            "Starting code generation",
            extra={
                "goal": action.goal,
                "mode": action.mode.value,
                "program_id": action.current_program.id if action.current_program else None,
                "instruction_count": len(action.instructions) if action.instructions else 0,
            },
        )

        try:
            # Build prompts
            system_prompt, user_prompt = self.prompt_builder.build_prompts(action)

            self.logger.debug(
                "Prompts built",
                extra={
                    "system_prompt_length": len(system_prompt),
                    "user_prompt_length": len(user_prompt),
                },
            )

            # Generate response from LLM
            response = await self.llm.generate(user_prompt, system_message=system_prompt)

            self.logger.debug(
                "LLM response received",
                extra={
                    "response_length": len(response),
                    "generation_time_ms": int((time.time() - start_time) * 1000),
                },
            )

            # Extract and process code
            result = self._extract_code_from_response(response, action.mode)

            self.logger.info(
                "Code generation completed successfully",
                extra={
                    "result_length": len(result),
                    "total_time_ms": int((time.time() - start_time) * 1000),
                },
            )

            return result

        except Exception as e:
            self.logger.error(
                "Code generation failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_time_ms": int((time.time() - start_time) * 1000),
                },
                exc_info=True,
            )
            raise

    def _extract_code_from_response(self, response: str, mode: EvolutionMode) -> str:
        """Extract final code from LLM response"""
        self.logger.debug(
            "Extracting code from response",
            extra={"mode": mode.value, "response_length": len(response)},
        )

        try:
            if mode == EvolutionMode.DIFF:
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

                    self.logger.debug(
                        "Diff extraction successful", extra={"change_count": len(extracted.changes)}
                    )

                    return changes_text
                else:
                    error_msg = f"Failed to extract diff changes: {extracted.error}"
                    self.logger.error("Diff extraction failed", extra={"error": extracted.error})
                    raise ValueError(error_msg)
            else:
                # Full rewrite
                extracted = self.program_extractor.extract_full_rewrite(response, self.language)
                if extracted.success:
                    new_code = extracted.program or response
                    # Ensure evolved code is enclosed in EVOLVE block comments
                   
                    self.logger.debug(
                        "Full rewrite extraction successful",
                        extra={"extracted_code_length": len(new_code)},
                    )

                    return new_code
                else:
                    error_msg = "Failed to extract code from full rewrite response"
                    self.logger.error(
                        "Full rewrite extraction failed",
                        extra={"error": extracted.error or "Unknown error"},
                    )
                    raise ValueError(error_msg)

        except Exception as e:
            self.logger.error(
                "Code extraction failed", extra={"error": str(e), "mode": mode.value}, exc_info=True
            )
            raise


