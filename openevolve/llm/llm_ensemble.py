"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional


from openevolve.llm.llm_interface import LLMInterface
from openevolve.llm.llm_openai import OpenAILLM


logger = logging.getLogger(__name__)


class EnsembleLLM(LLMInterface):
    """Ensemble of LLMs"""

    def __init__(self, 
                 ensemble_models: List[LLMInterface],
                 weights: Optional[List[float]] = None):
        self.ensemble_models = ensemble_models
        if weights is None:
            # Default to equal weights if not provided
            self.weights = [1.0] * len(ensemble_models)
        else:
            if len(weights) != len(ensemble_models):
                raise ValueError("Weights must match the number of models in the ensemble")
            self.weights = weights

        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        
        logger.info(
            f"Initialized LLM ensemble with {len(ensemble_models)} models with weights: {self.weights}"
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.generate(prompt, **kwargs)

    async def generate_one_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        return await model.generate_with_context(system_message, messages, **kwargs)

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(
            range(len(self.ensemble_models)), weights=self.weights, k=1
        )[0]
        sampled_model = self.ensemble_models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""

        # print the messages for debugging
        logger.debug(f"Generating with context: {system_message}, messages: {messages}")

        tasks = [
            model.generate_with_context(system_message, messages, **kwargs)
            for model in self.ensemble_models
        ]
        results = await asyncio.gather(*tasks)

        return results
