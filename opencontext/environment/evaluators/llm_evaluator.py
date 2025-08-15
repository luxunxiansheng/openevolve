"""
LLM-based evaluator for program assessment

Simplified implementation without prompt_sampler dependency.
"""

import json
import logging
import re
from typing import Dict

from opencontext.llm.llm_interface import LLMInterface
from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class LLMEvaluator(BaseEvaluator):
    """
    Evaluator that uses Large Language Models for program assessment

    Simplified implementation that builds prompts directly without PromptSampler.
    """

    def __init__(self, llm: LLMInterface):
        """
        Initialize the LLM evaluator

        Args:
                llm: Language model interface for evaluation
        """
        self.llm = llm

    async def evaluate(self, code: str, language: str = "python", **kwargs) -> Dict[str, float]:
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
            system_prompt = self._build_system_prompt()

            # Get LLM response
            responses = await self.llm.generate(prompt=user_prompt, system_message=system_prompt)

            # Parse responses and extract metrics
            avg_metrics = {}
            json_pattern = r"```json\n(.*?)\n```"

            for i, response in enumerate(responses):
                try:
                    # Try to extract JSON from markdown code blocks first
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to extract JSON directly
                        json_str = response
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
                    logger.warning(f"Error parsing JSON from response {i}: {str(e)}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing response {i}: {str(e)}")
                    continue

            return avg_metrics

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            raise

    def _build_evaluation_prompt(self, code: str, language: str) -> str:
        """Build evaluation prompt directly"""
        return f"""Please evaluate this {language} code and provide a JSON response with numeric scores (0.0 to 1.0) for the following criteria:\n\nCode to evaluate:\n```{language}\n{code}\n```\n\nEvaluate the code on these criteria:\n- readability: How clear and understandable the code is\n- maintainability: How easy it is to modify and extend\n- efficiency: How performant and optimized the code is\n- style: How well it follows coding conventions\n- correctness: How likely the code is to work correctly\n\nPlease respond with a JSON object containing only these numeric scores, like:\n```json\n{{\n    \"readability\": 0.8,\n    \"maintainability\": 0.7,\n    \"efficiency\": 0.9,\n    \"style\": 0.8,\n    \"correctness\": 0.9\n}}\n```"""

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM evaluation"""
        return """You are an expert code reviewer. Evaluate code objectively and provide numeric scores between 0.0 and 1.0 for each criterion. Always respond with valid JSON format."""
