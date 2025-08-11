# we will use ray job client to submit the evaluation job
import os
import re
import tempfile
import time
import logging
import ast

from ray.job_submission import JobSubmissionClient, JobStatus

from openevolve.critic.critic import EvaluationResult, Critic

logger = logging.getLogger(__name__)


class PythonExecutionCritic(Critic):
    def __init__(
        self,
        critic_program_path: str,
        ray_head_ip: str = "http://127.0.0.1:8265",
        job_timeout_seconds: int = 300,
        status_check_interval: int = 5,
        job_stop_wait_time: int = 30,
        deletion_wait_time: int = 2,
    ) -> None:
        self.critic_program_path = critic_program_path
        self.job_client = JobSubmissionClient(ray_head_ip)
        self.job_timeout_seconds = job_timeout_seconds
        self.status_check_interval = status_check_interval
        self.job_stop_wait_time = job_stop_wait_time
        self.delteion_wait_time = deletion_wait_time

        self.critic_program = None

    async def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Critic method to evaluate and log the metrics and artifacts of the given program code.

        Args:
            **kwargs: Must include 'python_code' and 'program_id'.
                     May include 'runtime_env' for custom execution environment.

        Returns:
            EvaluationResult containing metrics and artifacts from execution
        """

        # Fetch required arguments at the top
        evolved_program_code = kwargs.get("evolved_program_code", "")
        if not evolved_program_code:
            raise ValueError("evolved_program_code must be provided for evaluation.")

        program_id = kwargs.get("program_id")
        if not program_id:
            raise ValueError("program_id must be provided for evaluation.")

        runtime_env = kwargs.get("runtime_env", {})
        if not runtime_env:
            logger.warning("No runtime environment provided, using default settings.")

        if self.critic_program is None:
            with open(self.critic_program_path, "r") as file:
                self.critic_program = file.read()
                if not self.critic_program.strip():
                    raise ValueError("Critic program is empty or not found at the specified path.")

        # Optionally, check types
        if not isinstance(evolved_program_code, str) or not isinstance(self.critic_program, str):
            raise TypeError("Both evolved_program_code and critic_program must be strings.")

        # Combine the evolved program code with the critic program as the job script
        job_script = evolved_program_code + self.critic_program

        # Submit and monitor job
        with tempfile.TemporaryDirectory() as temp_dir:
            submission_id = self._submit_evaluation_job(
                job_script, program_id, runtime_env, temp_dir
            )

            # Wait for job completion and get results
            log_output = self._wait_for_job_completion(submission_id)
            result = self._extract_evaluation_result_from_logs(log_output)

            logger.info(f"Extracted evaluation result: {result}")
            return result

    def _submit_evaluation_job(
        self, python_code: str, program_id: str, runtime_env: dict, temp_dir: str
    ) -> str:
        """Submit the evaluation job to Ray cluster."""
        # Create Python file in temp directory
        python_file_path = os.path.join(temp_dir, "dynamic_job_script.py")
        with open(python_file_path, "w") as f:
            f.write(python_code)

        # Set up runtime environment
        runtime_env.setdefault("working_dir", temp_dir)

        logger.info(
            f"Submitting evaluation job with Python file: {python_file_path} "
            f"and program ID: {program_id} with runtime environment: {runtime_env}"
        )

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
                job_result = self.job_client.get_job_info(submission_id)
                logger.info(f"Job result: {job_result}")
                break
            elif status == JobStatus.FAILED:
                logger.error(f"Job {submission_id} failed.")
                break
            elif status == JobStatus.PENDING:
                logger.info(f"Job {submission_id} is pending.")
            elif status == JobStatus.RUNNING:
                logger.info(f"Job {submission_id} is running.")
            else:
                logger.warning(f"Job {submission_id} is in an unknown state: {status}")
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
        logger.info(f"Job {submission_id} logs:\n{log_output}")
        return log_output

    def _preprocess_log_line(self, line):
        # Replace np.float64(x) with x
        line = re.sub(r"np\.float64\(([^)]+)\)", r"\1", line)
        # Replace array([...]) with [...], works for 1D and 2D arrays - more comprehensive regex
        line = re.sub(r"array\((\[.*?\])\)", r"\1", line, flags=re.DOTALL)
        return line

    def _extract_evaluation_result_from_logs(self, log_output: str) -> EvaluationResult:
        """
        Extract evaluation results from job logs by parsing various log patterns.

        Args:
            log_output: Raw log output from the job execution

        Returns:
            EvaluationResult containing parsed metrics and artifacts
        """
        metrics = {}
        artifacts = {}

        # Simple error detection - just check for errors and store the whole log
        if self._has_execution_errors(log_output):
            artifacts["execution_error"] = log_output  # Store the entire log as error info
            metrics["execution_success"] = False
            metrics["error_score"] = 0.0  # Penalty for execution errors
        else:
            metrics["execution_success"] = True

        # Define regex patterns for different log formats
        patterns = self._get_log_patterns()
        lines = log_output.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            processed = False

            # Try each pattern type
            for pattern_name, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    i = self._process_log_match(pattern_name, match, lines, i, metrics, artifacts)
                    processed = True
                    break

            if not processed:
                i += 1

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def _has_execution_errors(self, log_output: str) -> bool:
        """
        Simple error detection - just check if there are any common error indicators.

        Args:
            log_output: Raw log output from job execution

        Returns:
            True if errors detected, False otherwise
        """
        error_indicators = [
            "Traceback (most recent call last):",
            "Error:",
            "Exception:",
            "FAILED",
            "Process terminated",
            "Segmentation fault",
            "Core dumped",
        ]

        return any(indicator in log_output for indicator in error_indicators)

    def _get_log_patterns(self) -> dict:
        """Get compiled regex patterns for log parsing."""
        return {
            "metric": re.compile(r"^Metric ([^:]+): (.+)$"),
            "artifact_str": re.compile(r"^Artifact ([^:]+): (.+)$"),
            "artifact_bytes": re.compile(r"^Artifact ([^:]+): (\d+) bytes$"),
            "eval_result": re.compile(r"Evaluation Result: (.*)"),
            "artifacts_dict": re.compile(r"Artifacts: (.*)"),
        }

    def _process_log_match(
        self,
        pattern_name: str,
        match,
        lines: list,
        current_index: int,
        metrics: dict,
        artifacts: dict,
    ) -> int:
        """
        Process a matched log pattern and update metrics/artifacts.

        Returns:
            Updated line index after processing
        """
        if pattern_name == "metric":
            return self._process_metric_line(match, metrics, current_index)
        elif pattern_name == "artifact_bytes":
            return self._process_artifact_bytes_line(match, artifacts, current_index)
        elif pattern_name == "artifact_str":
            return self._process_artifact_string_line(match, lines, artifacts, current_index)
        elif pattern_name == "eval_result":
            return self._process_evaluation_result_line(match, metrics, current_index)
        elif pattern_name == "artifacts_dict":
            return self._process_artifacts_dict_line(match, lines, artifacts, current_index)
        else:
            return current_index + 1

    def _process_metric_line(self, match, metrics: dict, index: int) -> int:
        """Process a single metric line."""
        key, value = match.group(1), match.group(2)
        try:
            # Try to parse as simple float first
            metrics[key] = float(value)
        except ValueError:
            # If not a simple number, try to parse as complex structure
            try:
                metrics[key] = self._parse_complex_metric_value(value)
            except Exception as e:
                logger.warning(f"Could not parse metric value '{value}' for key '{key}': {e}")
                # Store as string if all parsing fails
                metrics[key] = value
        return index + 1

    def _process_artifact_bytes_line(self, match, artifacts: dict, index: int) -> int:
        """Process a bytes artifact line."""
        key, size = match.group(1), int(match.group(2))
        artifacts[key] = b"\x00" * size
        return index + 1

    def _process_artifact_string_line(self, match, lines: list, artifacts: dict, index: int) -> int:
        """Process a string artifact line, handling multi-line arrays."""
        key, value = match.group(1), match.group(2)

        if value.startswith("["):
            # Multi-line array - collect all lines until closing bracket
            array_lines, new_index = self._collect_array_lines(lines, index, value)
            try:
                artifacts[key] = self._parse_array_from_lines(array_lines)
            except Exception as e:
                logger.warning(f"Failed to parse artifact array for {key}: {e}")
                artifacts[key] = value  # Fallback to string
            return new_index
        else:
            artifacts[key] = value
            return index + 1

    def _process_evaluation_result_line(self, match, metrics: dict, index: int) -> int:
        """Process an evaluation result line."""
        result_str = match.group(1)

        try:
            # Try ast.literal_eval first (fastest for simple cases)
            eval_str = self._preprocess_log_line(result_str)
            eval_dict = ast.literal_eval(eval_str)
            metrics.update(eval_dict)
        except Exception as e:
            logger.debug(f"Could not parse Evaluation Result dict: {e}")
            # Fallback to manual extraction
            try:
                metrics.update(self._extract_simple_metrics(result_str))
            except Exception as e2:
                logger.warning(f"Failed to extract any metrics from evaluation result: {e2}")

        return index + 1

    def _process_artifacts_dict_line(self, match, lines: list, artifacts: dict, index: int) -> int:
        """Process a multi-line artifacts dictionary."""
        # Collect all lines until closing brace
        artifact_dict_lines, new_index = self._collect_dict_lines(lines, index, match.group(1))
        artifacts_str = "\n".join(artifact_dict_lines)

        try:
            # Try comprehensive parsing first
            artifacts.update(self._parse_artifacts_dict(artifacts_str))
        except Exception as e:
            logger.warning(f"Failed to parse Artifacts dict: {e}")
            # Fallback to simple extraction
            try:
                artifacts.update(self._extract_simple_artifacts(artifacts_str))
            except Exception as e2:
                logger.warning(f"Failed to extract any artifacts: {e2}")

        return new_index

    def _collect_array_lines(self, lines: list, start_index: int, first_value: str) -> tuple:
        """Collect lines for a multi-line array until closing bracket."""
        array_lines = [first_value]
        i = start_index

        while not array_lines[-1].strip().endswith("]") and i + 1 < len(lines):
            i += 1
            array_lines.append(lines[i])

        return array_lines, i + 1

    def _collect_dict_lines(self, lines: list, start_index: int, first_line: str) -> tuple:
        """Collect lines for a multi-line dictionary until closing brace."""
        dict_lines = [first_line]
        brace_count = first_line.count("{") - first_line.count("}")
        i = start_index

        while brace_count > 0 and i + 1 < len(lines):
            i += 1
            dict_lines.append(lines[i])
            brace_count += lines[i].count("{") - lines[i].count("}")

        return dict_lines, i + 1

    def _parse_array_from_lines(self, array_lines: list) -> list:
        """Parse array values from collected lines."""
        array_str = "\n".join(array_lines)
        array_str_clean = (
            array_str.replace("[", "").replace("]", "").replace("\n", " ").replace(",", " ")
        )
        return [float(x) for x in array_str_clean.split() if x]

    def _handle_existing_job(self, job_id: str) -> None:
        """
        Check if a job with the given ID already exists and handle it appropriately.
        Cancel/stop running jobs, or delete completed ones.

        Args:
            job_id: ID of the job to check and handle
        """
        try:
            existing_status = self.job_client.get_job_status(job_id)
            logger.info(f"Found existing job {job_id} with status: {existing_status}")

            # Stop running or pending jobs
            if existing_status in [JobStatus.PENDING, JobStatus.RUNNING]:
                logger.info(f"Stopping existing job {job_id} before submitting new one.")
                self.job_client.stop_job(job_id)
                self._wait_for_job_to_stop(job_id)

            # Delete the job to allow reuse of the submission_id
            logger.info(f"Deleting existing job {job_id} to reuse submission_id")
            self.job_client.delete_job(job_id)
            time.sleep(self.delteion_wait_time)  # Wait for deletion to complete

        except Exception as e:
            # Job might not exist, which is fine for new submissions
            logger.debug(f"No existing job found with ID {job_id}: {e}")

    def _wait_for_job_to_stop(self, job_id: str) -> None:
        """Wait for a job to stop within the timeout period."""
        wait_time = 0
        while wait_time < self.job_stop_wait_time:
            status = self.job_client.get_job_status(job_id)
            if status in [JobStatus.STOPPED, JobStatus.FAILED, JobStatus.SUCCEEDED]:
                logger.info(f"Job {job_id} stopped with status: {status}")
                return
            time.sleep(1)
            wait_time += 1

        logger.warning(f"Job {job_id} did not stop within {self.job_stop_wait_time} seconds")

    def _parse_artifacts_dict(self, artifacts_str: str) -> dict:
        """Parse artifacts dictionary from string."""
        try:
            # Try simple ast.literal_eval first
            processed_str = self._preprocess_log_line(artifacts_str)
            return ast.literal_eval(processed_str)
        except Exception as e:
            logger.warning(f"Failed to parse artifacts dict: {e}")
            return {}

    def _extract_simple_metrics(self, metrics_str: str) -> dict:
        """Extract simple metrics from string."""
        try:
            processed_str = self._preprocess_log_line(metrics_str)
            return ast.literal_eval(processed_str)
        except Exception as e:
            logger.warning(f"Failed to parse metrics: {e}")
            return {}

    def _extract_simple_artifacts(self, artifacts_str: str) -> dict:
        """Extract simple artifacts from string."""
        try:
            processed_str = self._preprocess_log_line(artifacts_str)
            return ast.literal_eval(processed_str)
        except Exception as e:
            logger.warning(f"Failed to parse artifacts: {e}")
            return {}

    def _parse_complex_metric_value(self, value_str: str):
        """Parse complex metric values."""
        try:
            return ast.literal_eval(value_str)
        except Exception:
            return value_str
