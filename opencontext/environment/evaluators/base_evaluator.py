"""
Base evaluator class for program evaluation in the environment
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseEvaluator(ABC):
	"""
	Abstract base class for program evaluators

	All evaluators should inherit from this class and implement the evaluate method.
	"""

	def __init__(self, name: Optional[str] = None):
		"""
		Initialize the evaluator

		Args:
			name: Optional name for the evaluator
		"""
		self.name = name or self.__class__.__name__

	@abstractmethod
	async def evaluate(self, code: str, language: str = "python", **kwargs) -> Dict[str, float]:
		"""
		Evaluate a program and return metrics

		Args:
			code: The program code to evaluate
			language: Programming language (e.g., "python", "java", "cpp")
			**kwargs: Additional parameters for evaluation

		Returns:
			Dictionary mapping metric names to float values
			Should always include a 'score' key with overall evaluation

		Raises:
			NotImplementedError: If not implemented by subclass
		"""
		pass

	def __str__(self) -> str:
		return f"{self.name}Evaluator"

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(name='{self.name}')"
