from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ActionResult:
    """Result that can be pickled and sent between processes"""

    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None


class Actor(ABC):
    """
    Abstract base class for actors.
    All actors should inherit from this class and implement the act method.
    """

    @abstractmethod
    async def act(self, **kwargs) -> None:
        pass  # Perform the action based on the provided parameters.
