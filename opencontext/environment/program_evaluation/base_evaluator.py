"""
Base evaluator class for program evaluation in the environment
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts

    This maintains backward compatibility with the existing dict[str, float] contract
    while adding a side-channel for arbitrary artifacts (text or binary data).
    """

    metrics: Dict[str, float]  # mandatory - existing contract
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)  # optional side-channel

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns for backward compatibility"""
        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, float]:
        """Backward compatibility - return just metrics"""
        return self.metrics

    def has_artifacts(self) -> bool:
        """Check if this result contains any artifacts"""
        return bool(self.artifacts)

    def get_artifact_keys(self) -> list:
        """Get list of artifact keys"""
        return list(self.artifacts.keys())

    def get_artifact_size(self, key: str) -> int:
        """Get size of a specific artifact in bytes"""
        if key not in self.artifacts:
            return 0

        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def get_total_artifact_size(self) -> int:
        """Get total size of all artifacts in bytes"""
        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())


class BaseEvaluator(ABC):
    """
    Abstract base class for program evaluators

    All evaluators should inherit from this class and implement the evaluate method.
    """

    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the evaluator

        Args:
            name: Optional name for the evaluator
            logger: Optional logger instance, creates default if not provided
        """
        self.name = name or self.__class__.__name__
        self.logger = logger or logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    async def evaluate(self, code: str, language: str = "python", **kwargs) -> EvaluationResult:
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

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log the metrics during evaluation. Uses logger if available, otherwise prints.

        Args:
            metrics: A dictionary containing metric names and their values.
        """
        for key, value in metrics.items():
            message = f"Metric {key}: {value}"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)

    def log_artifact(self, artifacts: Dict[str, Union[str, bytes]]) -> None:
        """
        Log an artifact during evaluation. Uses logger if available, otherwise prints.

        Args:
            artifacts: A dictionary containing artifact keys and their values.
        """
        for key, value in artifacts.items():
            if isinstance(value, str):
                message = f"Artifact {key}: {value}"
            elif isinstance(value, bytes):
                message = f"Artifact {key}: {len(value)} bytes"
            else:
                message = f"Artifact {key}: {str(value)}"

            if self.logger:
                self.logger.info(message)
            else:
                print(message)
