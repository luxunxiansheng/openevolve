from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Union


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


class Evaluator(ABC):
    """
    Abstract base class for evaluators.
    All evaluators should inherit from this class and implement the evaluate method.
    """
    @abstractmethod 
    def evaluate(self,**kwargs) -> EvaluationResult:
        pass # evaluate and log the metrics and artifacts of the given program code.

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log the metrics during evaluation. The defalut implementation is to print the metrics.
        :param metrics: A dictionary containing metric names and their values.
        :return: None

        """
        for key, value in metrics.items():
            print(f"Metric {key}: {value}")
        
    
    
    def log_artifact(self, artifacts:Dict[str, Union[str, bytes]]) -> None:
        """
        Log an artifact during evaluation. The default implementation is to print the artifact 


        :param artifacts: A dictionary containing artifact keys and their values.
        :return: None
        
        """
        for key, value in artifacts.items():
            if isinstance(value, str):
                print(f"Artifact {key}: {value}")
            elif isinstance(value, bytes):
                print(f"Artifact {key}: {len(value)} bytes")
            else:
                print(f"Artifact {key}: {str(value)}")



        
        