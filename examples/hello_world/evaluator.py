from opencontext.critic.critic import EvaluationResult, Critic


class HelloWorldEvaluator(Critic):
    """
    A simple evaluator that returns a fixed evaluation result.
    This is just for demonstration purposes.
    """

    async def evaluate(self, **kwargs) -> EvaluationResult:

        # Simulate evaluation logic
        metrics = {"execution_time": 0.1, "memory_usage": 10.5}
        artifacts = {"output": "Hello, World!"}

        self.log_artifact(artifacts)
        self.log_metrics(metrics)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)


import asyncio

if __name__ == "__main__":
    evaluator = HelloWorldEvaluator()
    asyncio.run(evaluator.evaluate(program_code="I am a simple evaluator that does nothing."))
