"""
LLM-based evaluator for program assessment

Simplified implementation without prompt_sampler dependency.
"""

import json
import logging
import re
from typing import Optional

from opencontext.llm.llm_interface import LLMInterface
from opencontext.environment.program_evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
)
from opencontext.environment.templates.template_manager import TemplateManager


class LLMEvaluator(BaseEvaluator):
    """
    Evaluator that uses Large Language Models for program assessment

    Simplified implementation that builds prompts directly without PromptSampler.
    """

    def __init__(self, llm: LLMInterface, logger: Optional[logging.Logger] = None):
        """
        Initialize the LLM evaluator

        Args:
                llm: Language model interface for evaluation
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        # Template manager for system prompts; templates live in environment/templates
        try:
            self.template_manager = TemplateManager()
        except Exception:
            self.template_manager = None

    async def evaluate(self, code: str, language: str = "python", **kwargs) -> EvaluationResult:
        """
        Evaluate program using LLM assessment with structured JSON responses

        Args:
                code: Program code to evaluate
                language: Programming language
                **kwargs: Additional parameters

        Returns:
                Dictionary of numeric metrics from LLM evaluation
        """
        if not code:
            raise ValueError("code must be provided for evaluation.")

        try:
            # Build prompt directly without PromptSampler
            user_prompt = self._build_evaluation_prompt(code, language)
            # Load system prompt from template if available
            try:
                if self.template_manager:
                    tmpl = self.template_manager.load_template("eval")
                    system_prompt = tmpl.format(program=code, language=language)
                else:
                    system_prompt = self._build_system_prompt()
            except Exception:
                system_prompt = self._build_system_prompt()

            # Get LLM response
            responses = await self.llm.generate(prompt=user_prompt, system_message=system_prompt)

            # Normalize responses: LLM client may return a single string or an iterable
            if not isinstance(responses, (list, tuple)):
                responses = [responses]

            # Parse responses and extract metrics
            avg_metrics = {}
            json_pattern = r"```json\n(.*?)\n```"

            for i, response in enumerate(responses):
                try:
                    # If the response is already a parsed dict, use it
                    if isinstance(response, dict):
                        result = response
                    else:
                        # Ensure we treat response as text
                        resp_text = str(response)
                        # Try to extract JSON from markdown code blocks first
                        json_match = re.search(json_pattern, resp_text, re.DOTALL)

                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            # Try to extract JSON directly
                            json_str = resp_text
                            # Remove non-JSON parts
                            start_idx = json_str.find("{")
                            end_idx = json_str.rfind("}") + 1
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = json_str[start_idx:end_idx]

                        # Parse JSON
                        result = json.loads(json_str)

                    # Extract only numeric metrics
                    metrics = {}
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value

                    # Use weights if available (EnsembleLLM), else default to 1.0
                    weight = (
                        getattr(self.llm, "weights", [1.0] * len(responses))[i]
                        if hasattr(self.llm, "weights")
                        else 1.0
                    )

                    # Average the metrics with weights
                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parsing JSON from response {i}: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error processing response {i}: {str(e)}")
                    continue

            # Wrap averaged metrics in EvaluationResult
            return EvaluationResult(metrics=avg_metrics)
        except Exception as e:
            self.logger.error(f"Error in LLM evaluation: {str(e)}")
            raise

    def _build_evaluation_prompt(self, code: str, language: str) -> str:
        """Builds the user-level evaluation prompt containing the task/instruction."""
        # Keep a concise instruction for the LLM to return JSON only
        return f"Please evaluate the following {language} program and return a JSON object with numeric metrics.\n\n{code}"

    def _build_system_prompt(self) -> str:
        """Fallback system prompt if template loading fails."""
        return (
            "You are an automated code evaluator. Assess the provided program and "
            "return a single JSON object with numeric scores for correctness, readability, "
            "maintainability, and performance. Include an optional 'notes' string."
        )
