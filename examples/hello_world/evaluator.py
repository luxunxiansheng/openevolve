

from openevolve.evaluation.evaluator import EvaluationResult, RayEvaluator


class HelloWorldEvaluator(RayEvaluator):
    """
    A simple evaluator that returns a fixed evaluation result.
    This is just for demonstration purposes.
    """
    def run(self, program_code: str, **kwargs) -> EvaluationResult:
        """
        Evaluate the given program code and return a fixed EvaluationResult.
        """
        # Simulate evaluation logic
        metrics = {"execution_time": 0.1, "memory_usage": 10.5}
        artifacts = {"output": "Hello, World!"}

        return EvaluationResult(metrics=metrics, artifacts=artifacts)


if __name__ == "__main__":
    evaluator = HelloWorldEvaluator()
    evaluator.evaluate("I am a simple evaluator that does nothing.")
    
       