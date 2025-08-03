import json
import logging
import re
import traceback

from openevolve.critique.critic import Critic, EvaluationResult
from openevolve.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class LLMCritic(Critic):
    """
    Critic that uses a large language model (LLM) to evaluate Python code.
    This critic can be used to assess the quality of code, provide feedback,
    and suggest improvements based on LLM capabilities.

    It's now regarded as an Agent.
    """

    def __init__(self, 
                 llm_client: LLMInterface,
                 prompt_sampler) -> None:
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
        program_code = kwargs.get("program_code")
        if not program_code:
            raise ValueError("program_code must be provided for evaluation.")
        
        program_id = kwargs.get("program_id", "default_program_id")
        if not isinstance(program_id, str):
            raise ValueError("program_id must be a string.")
        
        try:
            # Create prompt for LLM
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )

            # Get LLM response
            responses = await self.llm_client.generate_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
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
                    
                    # Weight of the model in the ensemble (default to 1.0 as ensemble is not defined)
                    weight = self.llm_client.weights[i] if self.llm_client.weights else 1.0

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




   