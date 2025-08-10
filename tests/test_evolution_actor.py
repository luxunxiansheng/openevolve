import unittest
import asyncio

import ray

from openevolve.actor.evolution_actor import EvolutionActor
from openevolve.database.database import ProgramDatabase, Program
from openevolve.prompt.sampler import PromptSampler
from openevolve.critic.llm_critic import LLMCritic
from openevolve.critic.exe_critic import PythonExecutionCritic
from openevolve.llm.llm_ensemble import EnsembleLLM
from openevolve.llm.llm_openai import OpenAILLM


# Initialize the execution critic
critic_file_path = (
            "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/critic.py"
        )

evovle_program_path = (
            "/workspaces/openevolve/examples/circle_packing_with_artifacts_new/circle_packing.py"
        )


class TestEvolutionActor(unittest.TestCase):
    def setUp(self):
        self.database = ray.remote(ProgramDatabase).remote()

        # Initialize the LLM client for the actor
        actor_prompt_sampler = PromptSampler(system_template_key="actor_system_message")
        llm_actor_client = EnsembleLLM([OpenAILLM(name="Qwen3-14B-AWQ")])

        critic_prompt_sampler = PromptSampler(system_template_key="evaluator_system_message")
        llm_critic_client = EnsembleLLM([OpenAILLM(name="Qwen3-14B-AWQ")])
        llm_critic = LLMCritic(llm_critic_client, critic_prompt_sampler)
       
        exe_critic = PythonExecutionCritic(critic_program_path=critic_file_path)

        # Initialize the EvolutionActor with the database and critics
        self.actor = EvolutionActor(
            database=self.database,
            actor_prompt_sampler=actor_prompt_sampler,
            llm_actor_client=llm_actor_client,
            llm_critic=llm_critic,
            exe_critic=exe_critic,
        )

        with open(evovle_program_path, "r") as file:
            python_code = file.read()

        parent_program = Program(
            id="parent1",
            code=python_code,
            language="python",
            parent_id=None,
            generation=0,
            metrics={},
            iteration_found=0,
            metadata={},
        )
        self.database.add.remote(parent_program)

    def test_evolution_actor_act_returns_result(self):
        async def run_act():
            result = await self.actor.act(iteration=0)
            print("Action Result:", result)
            self.assertIsNotNone(result)

        asyncio.run(run_act())


if __name__ == "__main__":
    unittest.main()
