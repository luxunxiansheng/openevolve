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

    def __init__(
        self,
        llm: LLMInterface,
        template_name: str = "program_eval",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the LLM evaluator

        Args:
            llm: Language model interface for evaluation
            template_name: Name of the template to use for evaluation
            logger: Optional logger instance
        """
        # Initialize parent class with logger
        super().__init__(name="LLMEvaluator", logger=logger)

        self.llm = llm
        self.template_manager = TemplateManager()
        self.template_name = template_name

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
            tmpl = self.template_manager.load_template(self.template_name)
            user_prompt = tmpl.format(program=code, language=language)

            # Get LLM response
            responses = await self.llm.generate(
                prompt=user_prompt,
                system_message="You are an expert program reviewer and evaluator with more than 15 years of experience in software development and code quality assessment.",
            )

            # Normalize responses: LLM client may return a single string or an iterable
            if not isinstance(responses, (list, tuple)):
                responses = [responses]

            # Parse responses and extract metrics
            avg_metrics = {}
            all_artifacts = {}
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

                    # Extract numeric values into metrics and non-numeric into artifacts
                    metrics = {}
                    artifacts = {}
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                        else:
                            # Convert non-numeric values to string artifacts
                            artifacts[key] = str(value)

                    # Store artifacts from this response (keyed by response index if multiple)
                    if artifacts:
                        if len(responses) > 1:
                            for key, value in artifacts.items():
                                all_artifacts[f"{key}_response_{i}"] = value
                        else:
                            all_artifacts.update(artifacts)

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
            return EvaluationResult(metrics=avg_metrics, artifacts=all_artifacts)
        except Exception as e:
            self.logger.error(f"Error in LLM evaluation: {str(e)}")
            raise
