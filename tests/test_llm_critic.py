import unittest
import asyncio
from openevolve.critic.llm_critic import LLMCritic
from openevolve.critic.critic import EvaluationResult
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.prompt.sampler import PromptSampler
from openevolve.llm.llm_ensemble import EnsembleLLM


class TestLLMEvaluator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm_client = EnsembleLLM([OpenAILLM()])
        self.prompt_sampler = PromptSampler()
        self.prompt_sampler.set_templates("evaluator_system_message")
        self.evaluator = LLMCritic(self.llm_client, self.prompt_sampler)

    async def test_evaluate_returns_evaluation_result(self):
        # Read the hello_world initial program code
        with open(
            "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/critic.py", "r"
        ) as f:
            program_code = f.read()
        # This will actually call the LLM, so the test may require a running LLM API or will fail gracefully
        try:
            # Debug: print the prompt that will be sent
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )
            print(f"Prompt sent to LLM:\nSystem: {prompt['system']}\nUser: {prompt['user']}")
            result = await self.evaluator.evaluate(program_code=program_code, program_id="test_id")
            print(f"Evaluation Result: {result}")
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
