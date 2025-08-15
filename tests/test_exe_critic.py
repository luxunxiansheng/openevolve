import unittest

# Adjust the import path as needed
from opencontext.critic.exe_critic import PythonExecutionCritic

critic_python_file_path = (
    "/workspaces/opencontext/examples/circle_packing_with_artifacts_new/critic.py"
)
python_evovled_file_path = (
    "/workspaces/opencontext/examples/circle_packing_with_artifacts_new/circle_packing.py"
)


class TestOrchestrator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):

        self.exe_critic = PythonExecutionCritic(critic_python_file_path)

    async def test_evaluate_python(self):
        # This is a placeholder for an actual test.
        # You would need to create a valid Python file and runtime environment for a real test.
        # Replace with an actual script path
        runtime_env = {}  # Adjust as needed

        with open(python_evovled_file_path, "r") as file:
            python_code = file.read()

        # Await the coroutine returned by evaluate
        result = await self.exe_critic.evaluate(
            evolved_program_code=python_code, runtime_env=runtime_env, program_id="test_program_4"
        )
        print(result)


if __name__ == "__main__":
    unittest.main()
