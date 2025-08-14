import asyncio
import unittest
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.llm.llm_ensemble import EnsembleLLM


class TestOpenAILLM(unittest.TestCase):
    def setUp(self):
        # Use OpenAILLM with default arguments
        self.llm = OpenAILLM()
        self.ensemble = EnsembleLLM([OpenAILLM(), OpenAILLM()])

    def test_ensemble_config(self):
        # Create an ensemble with two models of different weights
        model1 = {"name": "Qwen3-14B-AWQ", "weight": 1.0}
        model2 = {"name": "Qwen3-14B-AWQ", "weight": 3.0}
        ensemble = [model1, model2]
        # Check that the ensemble contains both models
        self.assertEqual(len(ensemble), 2)
        # Check that weights are normalized correctly
        total_weight = sum(m["weight"] for m in ensemble)
        normalized = [m["weight"] / total_weight for m in ensemble]
        self.assertAlmostEqual(normalized[0], 0.25)
        self.assertAlmostEqual(normalized[1], 0.75)
        # Check that the weights property matches the model weights
        self.assertEqual([m["weight"] for m in ensemble], [1.0, 3.0])

    def test_generate(self):
        async def run():
            result = await self.llm.generate("can you program a python app?")
            print("Result:", result)
            self.assertIsInstance(result, str)

        asyncio.run(run())

    def test_ensemble_generate(self):
        async def run():
            result = await self.ensemble.generate("can you program a python app?")
            print("Ensemble Result:", result)
            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(r, str) for r in result))
            # Optionally, check the first result
            self.assertIsInstance(result[0], str)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
