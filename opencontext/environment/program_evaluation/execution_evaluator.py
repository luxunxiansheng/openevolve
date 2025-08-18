"""
Execution Evaluator for OpenContext Environment

Based on the exe_critic implementation with Ray cluster support for distributed execution.
"""

import logging
import os
import tempfile
import time
import uuid
from typing import Dict, Union, Optional

from ray.job_submission import JobSubmissionClient, JobStatus

from opencontext.environment.program_evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
)

logger = logging.getLogger(__name__)


class ExecutionEvaluator(BaseEvaluator):
    """
    Evaluator for executing and measuring program performance using Ray cluster.

    This evaluator submits code execution jobs to a Ray cluster for safe, distributed evaluation.
    """

    def __init__(
        self,
        critic_program_path: str,
        ray_head_ip: str = "http://127.0.0.1:8265",
        job_timeout_seconds: int = 30,
        status_check_interval: int = 1,
        job_stop_wait_time: int = 30,
        deletion_wait_time: int = 2,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize ExecutionEvaluator with critic program and Ray cluster configuration.

        Args:
                critic_program_path: Path to the critic program file (e.g., critic.py)
                ray_head_ip: Ray cluster head node IP address
                job_timeout_seconds: Maximum time to wait for job completion
                status_check_interval: Interval between status checks in seconds
                job_stop_wait_time: Time to wait for job stop
                deletion_wait_time: Time to wait before deletion
                logger: Optional logger instance
        """
        # Initialize parent class with logger
        super().__init__(name="ExecutionEvaluator", logger=logger)

        self.critic_program_path = critic_program_path
        self.ray_head_ip = ray_head_ip
        self.job_timeout_seconds = job_timeout_seconds
        self.status_check_interval = status_check_interval
        self.job_stop_wait_time = job_stop_wait_time
        self.deletion_wait_time = deletion_wait_time

        # Initialize Ray job client
        self.job_client = JobSubmissionClient(ray_head_ip)

        # Cache for critic program content
        self.critic_program = None

        self.logger.info(
            f"ExecutionEvaluator initialized with critic: {critic_program_path}, Ray head: {ray_head_ip}"
        )

    async def evaluate(self, code: str, language: str = "python", **kwargs) -> EvaluationResult:
        """
        Evaluate program by combining it with critic program and submitting to Ray cluster.

        Args:
                code: Evolved program code to evaluate
                language: Programming language (currently supports 'python')
                **kwargs: Additional evaluation parameters (program_id, runtime_env, etc.)

        Returns:
                Dictionary containing evaluation metrics parsed from critic program output
        """
        try:
            # Generate program_id if not provided
            program_id = kwargs.get("program_id", f"eval_{uuid.uuid4().hex[:8]}")
            runtime_env = kwargs.get("runtime_env", {})

            # Load critic program if not cached
            if self.critic_program is None:
                with open(self.critic_program_path, "r") as file:
                    self.critic_program = file.read()
                    if not self.critic_program.strip():
                        raise ValueError(
                            f"Critic program is empty or not found at: {self.critic_program_path}"
                        )

            # Validate inputs
            if not isinstance(code, str) or not isinstance(self.critic_program, str):
                raise TypeError("Both evolved code and critic_program must be strings.")

            # Combine the evolved program code with the critic program as the job script
            job_script = code + "\n\n" + self.critic_program

            # Submit and monitor job
            with tempfile.TemporaryDirectory() as temp_dir:
                submission_id = self._submit_evaluation_job(
                    job_script, program_id, runtime_env, temp_dir
                )

                # Wait for job completion and get results
                log_output = self._wait_for_job_completion(submission_id)
                metrics, artifacts = self._extract_metrics_and_artifacts_from_logs(log_output)

                logger.info(f"Extracted {len(metrics)} metrics: {list(metrics.keys())}")
                logger.info(f"Extracted {len(artifacts)} artifacts: {list(artifacts.keys())}")
                logger.debug(f"Metrics values: {metrics}")
                if artifacts:
                    logger.debug(
                        f"Artifacts preview: {[(k, str(v)[:100] + '...' if len(str(v)) > 100 else str(v)) for k, v in artifacts.items()]}"
                    )

                # Wrap metrics and artifacts in EvaluationResult
                return EvaluationResult(metrics=metrics, artifacts=artifacts)

        except Exception as e:
            logger.error(f"Execution evaluation failed: {e}")
            raise

    def _submit_evaluation_job(
        self, python_code: str, program_id: str, runtime_env: dict, temp_dir: str
    ) -> str:
        """Submit the evaluation job to Ray cluster."""
        # Create Python file in temp directory
        python_file_path = os.path.join(temp_dir, f"eval_{program_id}.py")
        with open(python_file_path, "w") as f:
            f.write(python_code)

        # Set up runtime environment
        runtime_env.setdefault("working_dir", temp_dir)

        logger.info(f"Submitting evaluation job {program_id} to Ray cluster")

        # Handle existing job with same ID
        self._handle_existing_job(program_id)

        # Submit new job
        entrypoint = f"python {python_file_path}"
        return self.job_client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            submission_id=program_id,
        )

    def _wait_for_job_completion(self, submission_id: str) -> str:
        """Wait for job to complete and return logs."""
        start_time = time.time()

        while True:
            status = self.job_client.get_job_status(submission_id)

            if status == JobStatus.SUCCEEDED:
                logger.info(f"Job {submission_id} completed successfully.")
                break
            elif status == JobStatus.FAILED:
                logger.error(f"Job {submission_id} failed.")
                break
            elif status in [JobStatus.PENDING, JobStatus.RUNNING]:
                logger.debug(f"Job {submission_id} status: {status}")
            else:
                logger.warning(f"Job {submission_id} in unknown state: {status}")
                break

            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.job_timeout_seconds:
                logger.error(
                    f"Job {submission_id} timed out after {self.job_timeout_seconds} seconds."
                )
                break

            time.sleep(self.status_check_interval)

        # Get and return logs
        log_output = self.job_client.get_job_logs(submission_id)
        logger.debug(f"Job {submission_id} logs:\n{log_output}")
        return log_output

    def _handle_existing_job(self, job_id: str) -> None:
        """Handle existing job with same ID by stopping and deleting it."""
        try:
            existing_status = self.job_client.get_job_status(job_id)
            logger.debug(f"Found existing job {job_id} with status: {existing_status}")

            # Stop running or pending jobs
            if existing_status in [JobStatus.PENDING, JobStatus.RUNNING]:
                logger.info(f"Stopping existing job {job_id}")
                self.job_client.stop_job(job_id)
                self._wait_for_job_to_stop(job_id)

            # Delete the job to allow reuse of submission_id
            logger.debug(f"Deleting existing job {job_id}")
            self.job_client.delete_job(job_id)

        except Exception:
            # Job doesn't exist, which is fine
            pass

    def _wait_for_job_to_stop(self, job_id: str, max_wait_time: int = 10) -> None:
        """Wait for job to stop before proceeding."""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                status = self.job_client.get_job_status(job_id)
                if status not in [JobStatus.PENDING, JobStatus.RUNNING]:
                    break
            except Exception:
                break
            time.sleep(0.5)

    def _extract_metrics_and_artifacts_from_logs(
        self, log_output: str
    ) -> tuple[Dict[str, float], Dict[str, Union[str, bytes]]]:
        """Extract evaluation metrics and artifacts from job logs using exe_critic patterns."""
        import re

        metrics = {}
        artifacts = {}

        try:
            # Parse metrics and artifacts using patterns similar to exe_critic
            lines = log_output.split("\n")

            for line in lines:
                line = line.strip()

                # Look for "Metric name: value" pattern from critic programs
                metric_match = re.match(r"^Metric\s+([^:]+):\s*(.+)$", line)
                if metric_match:
                    key, value = metric_match.groups()
                    try:
                        metrics[key.strip()] = float(value.strip())
                        logger.debug(f"Extracted metric: {key.strip()} = {value.strip()}")
                    except ValueError:
                        logger.warning(f"Could not parse metric value '{value}' for key '{key}'")
                        continue

                # Look for "Artifact name: value" pattern for non-numeric data
                artifact_match = re.match(r"^Artifact\s+([^:]+):\s*(.+)$", line)
                if artifact_match:
                    key, value = artifact_match.groups()
                    artifacts[key.strip()] = value.strip()
                    logger.debug(f"Extracted artifact: {key.strip()} = {value.strip()[:50]}...")

                # Also handle direct key: value patterns
                elif ":" in line and not line.startswith(
                    ("INFO", "DEBUG", "WARNING", "ERROR", "Artifact")
                ):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key, value = parts
                        key = key.strip().lower()
                        try:
                            # Try to parse as numeric metric first
                            metrics[key] = float(value.strip())
                            logger.debug(f"Extracted metric: {key} = {value.strip()}")
                        except ValueError:
                            # If not numeric, treat as artifact
                            artifacts[key] = value.strip()
                            logger.debug(f"Extracted artifact: {key} = {value.strip()[:50]}...")
                            continue

            logger.info(f"Total extraction: {len(metrics)} metrics, {len(artifacts)} artifacts")
            return metrics, artifacts

        except Exception as e:
            logger.error(f"Failed to extract metrics and artifacts from logs: {e}")
            raise
