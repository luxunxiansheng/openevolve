import asyncio
import unittest
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.llm.config import LLMConfig

from openevolve.llm.config import EnsembleLLMConfig

class TestOpenAILLM(unittest.TestCase):
    def setUp(self):
        # Use OpenAILLM with a default LLMConfig object
        self.llm = OpenAILLM(LLMConfig())

    def test_ensemble_config(self):
        # Create an ensemble with two models of different weights
        model1 = LLMConfig(name="Qwen3-14B-AWQ", weight=1.0)
        model2 = LLMConfig(name="Qwen3-14B-AWQ", weight=3.0)
        ensemble = EnsembleLLMConfig(ensemble_models=[model1, model2])
        # Check that the ensemble contains both models
        self.assertEqual(len(ensemble.ensemble_models), 2)
        # Check that weights are normalized correctly
        normalized = ensemble.get_normalized_weights()
        self.assertAlmostEqual(normalized[0], 0.25)
        self.assertAlmostEqual(normalized[1], 0.75)
        # Check that the weights property matches the model weights
        self.assertEqual(ensemble.weights, [1.0, 3.0])

    def test_generate(self):
        async def run():
            result = await self.llm.generate("can you program a python app?")
            print("Result:", result)
            self.assertIsInstance(result, str)
        asyncio.run(run())

    def test_generate_with_context(self):
        async def run():
            result = await self.llm.generate_with_context(
                system_message="Test system",
                messages=[{"role": "user", "content": "can you program a python app"}]
            )
            print("Result:", result)
            self.assertIsInstance(result, str)
        asyncio.run(run())

if __name__ == "__main__":
    unittest.main()
