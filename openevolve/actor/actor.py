from abc import abstractmethod, ABC
from dataclasses import dataclass


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None
    iteration: int = None

class Actor(ABC):
    """
    Abstract base class for actors.
    All actors should inherit from this class and implement the act method.
    """
    @abstractmethod
    async def act(self, **kwargs) -> None:
        pass  # Perform the action based on the provided parameters.
