"""
Ray-based task pool for distributed execution
"""

import logging
from typing import Any, Callable, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RayTaskPool:
    """
    A simple Ray-native task pool for distributed execution
    """

    def __init__(self, max_concurrency: int = 10):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not available. Install with: uv add ray")

        self.max_concurrency = max_concurrency
        self.tasks: List[ray.ObjectRef] = []

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> ray.ObjectRef:
        """
        Submit a function for execution

        Args:
            func: Function to run
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Ray ObjectRef
        """
        # Convert function to Ray remote if needed
        if not hasattr(func, "remote"):
            remote_func = ray.remote(func)
        else:
            remote_func = func

        task = remote_func.remote(*args, **kwargs)
        self.tasks.append(task)
        return task

    def get_results(self, timeout: Optional[float] = None) -> List[Any]:
        """
        Get results from all submitted tasks

        Args:
            timeout: Timeout in seconds

        Returns:
            List of task results
        """
        if not self.tasks:
            return []

        results = ray.get(self.tasks, timeout=timeout)
        self.tasks.clear()
        return results

    def cancel_all(self) -> None:
        """Cancel all pending tasks"""
        for task in self.tasks:
            ray.cancel(task)
        self.tasks.clear()

    def wait(self, num_returns: int = 1, timeout: Optional[float] = None) -> tuple:
        """
        Wait for a specific number of tasks to complete

        Args:
            num_returns: Number of tasks to wait for
            timeout: Timeout in seconds

        Returns:
            Tuple of (ready_tasks, remaining_tasks)
        """
        return ray.wait(self.tasks, num_returns=num_returns, timeout=timeout)
