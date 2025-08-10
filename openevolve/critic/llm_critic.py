import json
import logging

import re
import traceback
import uuid

from openevolve.critic.critic import Critic, EvaluationResult
from openevolve.llm.llm_interface import LLMInterface
from ..prompt.sampler import PromptSampler
from ..prompt.templates import Templates

logger = logging.getLogger(__name__)


class LLMCritic(Critic):
    """
    Critic that uses a large language model (LLM) to evaluate Python code.
    This critic can be used to assess the quality of code, provide feedback,
    and suggest improvements based on LLM capabilities.

    It's now regarded as an Agent.
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        prompt_sampler: PromptSampler,
    ) -> None:
        """
        Initialize the LLM critic with a client that interacts with the LLM.

        Args:
            llm_client (Any): An instance of a client that can communicate with the LLM.
        """
        self.llm_client = llm_client
        self.prompt_sampler = prompt_sampler

    async def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Evaluate the provided Python code using the LLM.

        Args:
            **kwargs: Arbitrary keyword arguments, including 'program_code' which is the code to evaluate.

        Returns:
            EvaluationResult: The result of the evaluation containing metrics and artifacts.
        """
        evolved_program_code = kwargs.get("evolved_program_code")
        if not evolved_program_code:
            raise ValueError("program_code must be provided for evaluation.")

        user_template_key = kwargs.get("user_template_key", Templates.CRITIC_SYSTEM)

        try:
            # Create prompt for LLM
            prompt = self.prompt_sampler.build_prompt(
                current_program=evolved_program_code, user_template_key=user_template_key
            )

            # Get LLM response
            responses = await self.llm_client.generate(
                prompt=prompt["user"], system_message=prompt["system"]
            )

            # Extract JSON from response
            artifacts = {}
            avg_metrics = {}
            json_pattern = r"```json\n(.*?)\n```"

            for i, response in enumerate(responses):
                try:
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

                    # Separate metrics (numeric) from artifacts (non-numeric)
                    metrics = {}
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                        else:
                            artifacts[key] = value

                    # Use weights if available (EnsembleLLM), else default to 1.0
                    weight = (
                        getattr(self.llm_client, "weights", [1.0] * len(responses))[i]
                        if hasattr(self.llm_client, "weights")
                        else 1.0
                    )

                    # Average the metrics
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

            return EvaluationResult(
                metrics=avg_metrics,
                artifacts=artifacts,
            )

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            traceback.print_exc()
            return EvaluationResult(metrics={}, artifacts={})
