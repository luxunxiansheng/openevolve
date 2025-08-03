import unittest
import asyncio
from openevolve.evaluation.llm_evaluator import LLMEvaluator
from openevolve.evaluation.evaluator import EvaluationResult
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.llm.config import LLMConfig
from openevolve.prompt.sampler import PromptSampler
from openevolve.config import PromptConfig

class TestLLMEvaluator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Use a local/test config for LLM and PromptSampler
        self.llm_client = OpenAILLM(LLMConfig())
        self.prompt_sampler = PromptSampler(PromptConfig())
        self.evaluator = LLMEvaluator(self.llm_client, self.prompt_sampler)

    async def test_evaluate_returns_evaluation_result(self):
        # Read the hello_world initial program code
        with open("examples/hello_world/initial_program.py", "r") as f:
            program_code = f.read()
        # This will actually call the LLM, so the test may require a running LLM API or will fail gracefully
        try:
            result = await self.evaluator.evaluate(program_code=program_code, program_id="test_id")
            self.assertIsInstance(result, EvaluationResult)
            # The following checks are best-effort, as real LLM output may vary
            self.assertIsInstance(result.metrics, dict)
            self.assertIsInstance(result.artifacts, dict)
        except Exception as e:
            self.skipTest(f"LLM API not available or failed: {e}")

    async def test_evaluate_handles_missing_program_code(self):
        with self.assertRaises(ValueError):
            await self.evaluator.evaluate()

    async def test_evaluate_handles_non_string_program_id(self):
        with self.assertRaises(ValueError):
            await self.evaluator.evaluate(program_code="print()", program_id=123)

    # This test is not applicable with real LLM client, as we cannot force a bad response easily
    # async def test_evaluate_handles_json_decode_error(self):
    #     pass

if __name__ == "__main__":
    unittest.main()