from typing import Any
from openevolve.evaluation.evaluator import EvaluationResult, Evaluator
from ..llm.large_language_model import LLMInterface


class LLMEvaluator(Evaluator):
    """
    Evaluator that uses a large language model (LLM) to evaluate Python code.
    This evaluator can be used to assess the quality of code, provide feedback,
    and suggest improvements based on LLM capabilities.

    It's now regarded as a Agent 
   
    """

    def __init__(self, 
                 llm_client: LLMInterface,
                 prompt_sampler) -> None:
        """
        Initialize the LLM evaluator with a client that interacts with the LLM.
        
        Args:
            llm_client (Any): An instance of a client that can communicate with the LLM.
        """
        self.llm_client = llm_client
        self.prompt_sampler = prompt_sampler

    async def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Evaluate the provided Python code using the LLM.
        
        Args:
            **kwargs: Arbitrary keyword arguments, including 'python_code' which is the code to evaluate.
        
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
            

            )
        
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate program {program_id}: {str(e)}")

        




   