import os
import asyncio
import unittest

from opencontext.environment.program_evaluation import LLMEvaluator
from opencontext.llm.llm_openai import OpenAILLM


class TestLLMEvaluatorIntegration(unittest.TestCase):
    def setUp(self):
        llm_client = OpenAILLM()
        self.evaluator = LLMEvaluator(llm=llm_client)

    def test_evaluate_returns_numeric_scores(self):
        code = """
def add(a, b):
    return a + b
"""
        # The evaluator is expected to return a mapping of metric->float between 0 and 1
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.evaluator.evaluate(code=code, language="python"))
        finally:
            loop.close()

        # Allow LLMEvaluator to return either a dataclass with `to_dict` or a dict
        metrics = (
            result.to_dict()
            if hasattr(result, "to_dict")
            else (result if isinstance(result, dict) else {})
        )
        self.assertIsInstance(metrics, dict)
        # At least one numeric metric should be present
        numeric_found = False
        for v in metrics.values():
            if isinstance(v, (int, float)):
                numeric_found = True
                break
        self.assertTrue(numeric_found, "LLMEvaluator did not return any numeric scores")


if __name__ == "__main__":
    unittest.main()
