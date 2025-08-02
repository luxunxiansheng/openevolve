"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List

from openevolve.llm.llm_interface import LLMInterface
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.config import LLMModelConfig

logger = logging.getLogger(__name__)


class EnsembleLLM(LLMInterface):
    """Ensemble of LLMs"""

    def __init__(self, ensemble_models_cfg: List[LLMModelConfig]):
        self.ensemble_models_cfg = ensemble_models_cfg

        # Initialize models from the configuration assume the models are OpenAI compatible
        self.ensemble_models = [OpenAILLM(model_cfg) for model_cfg in ensemble_models_cfg]

        # Extract and normalize model weights
        self.weights = [model.weight for model in ensemble_models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            ensemble_models_cfg
            and hasattr(ensemble_models_cfg[0], "random_seed")
            and ensemble_models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(ensemble_models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {ensemble_models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(ensemble_models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(ensemble_models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True
        
        self.generate_all_with_context = False

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
        index = self.random_state.choices(range(len(self.ensemble_models)), weights=self.weights, k=1)[0]
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

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        tasks = [
            model.generate_with_context(system_message, messages, **kwargs)
            for model in self.ensemble_models
        ]
        results = await asyncio.gather(*tasks)

        # Combine results from all models
        combined_result = "\n".join(results)
        logger.info(f"Combined result from all models: {combined_result}")
        return combined_result

    
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        if self.generate_all_with_context:
            return await self.generate_all_with_context(system_message, messages, **kwargs)
        else:
            return await self.generate_one_with_context(system_message, messages, **kwargs)
