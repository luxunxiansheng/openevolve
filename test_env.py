"""
Test script for ProgramEvolutionEnv
"""

import asyncio
import logging
from openevolve.environment.program_evolution_env import ProgramEvolutionEnv
from openevolve.llm.llm_openai import OpenAILLM
from openevolve.critic.exe_critic import PythonExecutionCritic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment():
    """Test the ProgramEvolutionEnv"""

    # Create LLM and critic
    llm = OpenAILLM(name="test-model", temperature=0.7, api_base="http://localhost:8010/v1")

    critic = ExecutionCritic(timeout=30)

    # Create environment
    env = ProgramEvolutionEnv(
        llm=llm,
        critic=critic,
        language="python",
        reward_scale=1.0,
        improvement_bonus=0.1,
        penalty_for_errors=-0.1,
    )

    logger.info("Created ProgramEvolutionEnv successfully!")

    # Test reset
    initial_program = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""

    observation, info = env.reset(options={"initial_program": initial_program})
    logger.info(f"Environment reset successfully")
    logger.info(f"Initial observation keys: {list(observation.keys())}")
    logger.info(f"Initial metrics: {info.get('initial_metrics', {})}")

    # Test step
    prompt = "Optimize this fibonacci function to use iteration instead of recursion for better performance"

    logger.info(f"Taking action with prompt: {prompt[:100]}...")
    obs, reward, terminated, truncated, info = env.step(prompt)

    logger.info(f"Step completed!")
    logger.info(f"Reward: {reward}")
    logger.info(f"Terminated: {terminated}, Truncated: {truncated}")
    logger.info(f"New metrics: {info.get('new_metrics', {})}")
    logger.info(f"Reward components: {info.get('reward_components', {})}")

    # Render environment
    env.render()

    # Close environment
    env.close()

    logger.info("Test completed successfully!")


if __name__ == "__main__":
    test_environment()
