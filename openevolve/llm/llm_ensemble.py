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

    def __init__(self, ensemble_models: List[LLMInterface], weights: Optional[List[float]] = None):
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

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(
            range(len(self.ensemble_models)), weights=self.weights, k=1
        )[0]
        sampled_model = self.ensemble_models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    async def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """Generate text using all available models and average their returned metrics"""
        randomly = kwargs.pop("randomly", False)
        logger.debug(
            f"Generating with prompt: {prompt}, system_message: {system_message}, randomly: {randomly}"
        )
        ensemble_models = self.ensemble_models if not randomly else [self._sample_model()]
        tasks = [
            model.generate(prompt, system_message=system_message, **kwargs)
            for model in ensemble_models
        ]
        results = await asyncio.gather(*tasks)
        return results
