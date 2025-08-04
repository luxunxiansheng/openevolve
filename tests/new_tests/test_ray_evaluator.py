import unittest

# Adjust the import path as needed
from openevolve.critic.exe_critic import PythonExecutionCritic


class TestOrchestrator(unittest.TestCase):
    def setUp(self):

        self.orchestrator = PythonExecutionCritic(ray_cluster_head_ip="http://localhost:8265")

    def test_evaluate_python(self):
        # This is a placeholder for an actual test.
        # You would need to create a valid Python file and runtime environment for a real test.
        python_file_path = "/workspaces/openevolve/examples/hello_world/evaluator.py"  # Replace with an actual script path
        runtime_env = {}  # Adjust as needed

        with open(python_file_path, "r") as file:
            python_code = file.read()
        

        # Uncomment the line below to run the actual evaluation (ensure the script exists)
        result = self.orchestrator.evaluate(
            python_code=python_code, runtime_env=runtime_env, program_id="test_program_2"
        )
        print(result)


if __name__ == "__main__":
    unittest.main()
