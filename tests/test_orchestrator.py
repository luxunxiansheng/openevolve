import unittest

# Adjust the import path as needed
from openevolve.evaluation.ray_evaluator import RayPythonEvaluationController

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
                
        self.orchestrator = RayPythonEvaluationController(ray_cluster_head_ip="http://localhost:8265")


    def test_evaluate_python(self):
        # This is a placeholder for an actual test.
        # You would need to create a valid Python file and runtime environment for a real test.
        python_file_path = "/workspaces/openevolve/examples/hello_world/evaluator.py"  # Replace with an actual script path
        runtime_env = {"working_dir": "/workspaces/openevolve/",}  # Adjust as needed

        # Uncomment the line below to run the actual evaluation (ensure the script exists)
        result = self.orchestrator.evaluate(python_file_path=python_file_path, runtime_env=runtime_env, program_id="test_program_1")
        print(result)
        

if __name__ == "__main__":
    unittest.main()

