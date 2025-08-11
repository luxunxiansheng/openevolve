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

        # Check for errors - simple detection
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

    def _extract_error_info(self, log_output: str) -> dict:
        """
        error_patterns = [
            # Python exceptions and errors
            r"Traceback \(most recent call last\):",
            r"^\w*Error:",
            r"^\w*Exception:",
            # System errors
            r"Segmentation fault",
            r"Core dumped",
            r"Process terminated",
            r"SIGKILL",
            r"SIGTERM",
            # Memory errors
            r"MemoryError",
            r"Out of memory",
            r"OOM",
            # Timeout errors
            r"TimeoutError",
            r"timeout exceeded",
            r"Process timeout",
            # Import/Module errors
            r"ModuleNotFoundError",
            r"ImportError",
            r"No module named",
            # Syntax errors
            r"SyntaxError",
            r"IndentationError",
            r"TabError",
            # Runtime errors specific to scientific computing
            r"ValueError.*array",
            r"TypeError.*numpy",
            r"LinAlgError",
            r"division by zero",
            # Ray-specific errors
            r"ray\.exceptions\.",
            r"WorkerCrashedError",
            r"RayTaskError",
        ]

        # Check for error patterns
        for pattern in error_patterns:
            if re.search(pattern, log_output, re.MULTILINE | re.IGNORECASE):
                error_info = self._extract_detailed_error_info(log_output, pattern)
                return True, error_info

        # Check for explicit failure indicators
        failure_indicators = [
            "FAILED",
            "ERROR:",
            "CRITICAL:",
            "Job failed",
            "Task failed",
            "Process crashed",
        ]

        for indicator in failure_indicators:
            if indicator in log_output:
                error_info = {
                    "error_type": "ExecutionFailure",
                    "error_message": f"Job failed with indicator: {indicator}",
                    "failure_indicator": indicator,
                    "detection_method": "failure_indicator",
                }
                return True, error_info

        # Check for incomplete execution (no success indicators)
        success_indicators = [
            "Evaluation Result:",
            "Metric ",
            "Artifact ",
            "completed successfully",
        ]

        has_success_indicator = any(indicator in log_output for indicator in success_indicators)
        if not has_success_indicator and log_output.strip():
            error_info = {
                "error_type": "IncompleteExecution",
                "error_message": "Job completed but produced no recognizable output",
                "log_snippet": log_output[-500:] if len(log_output) > 500 else log_output,
                "detection_method": "missing_success_indicators",
            }
            return True, error_info

        return False, {}

    def _extract_detailed_error_info(self, log_output: str, matched_pattern: str) -> dict:
        """
        Extract detailed error information based on the matched pattern.

        Args:
            log_output: Raw log output containing errors
            matched_pattern: The regex pattern that matched

        Returns:
            Dictionary containing detailed error information
        """
        error_info = {
            "detection_method": "pattern_match",
            "matched_pattern": matched_pattern,
        }

        lines = log_output.splitlines()

        # Extract Python traceback information
        if "Traceback" in matched_pattern:
            error_info.update(self._parse_python_traceback(lines))

        # Extract specific error types
        elif "Error:" in matched_pattern or "Exception:" in matched_pattern:
            error_info.update(self._parse_error_exception_line(log_output))

        # Extract numpy/scientific computing errors
        elif "numpy" in matched_pattern or "array" in matched_pattern:
            error_info.update(self._parse_numpy_error(log_output))

        # Extract Ray-specific errors
        elif "ray" in matched_pattern.lower():
            error_info.update(self._parse_ray_error(log_output))

        # Generic error extraction
        else:
            error_info.update(self._parse_generic_error(log_output, matched_pattern))

        # Always include a snippet of the error context
        error_info["error_context"] = self._extract_error_context(log_output)

        return error_info

    def _parse_python_traceback(self, lines: list) -> dict:
        """Parse Python traceback for detailed error information."""
        error_info = {}

        for i, line in enumerate(lines):
            if "Traceback (most recent call last):" in line:
                # Find the actual error message (last non-empty line after traceback)
                for j in range(len(lines) - 1, i, -1):
                    if lines[j].strip():
                        error_line = lines[j].strip()
                        if ":" in error_line:
                            error_parts = error_line.split(":", 1)
                            error_info["error_type"] = error_parts[0].strip()
                            error_info["error_message"] = error_parts[1].strip()
                        else:
                            error_info["error_message"] = error_line
                        break

                # Extract file and line information from traceback
                file_info = []
                for k in range(i + 1, len(lines)):
                    if lines[k].strip().startswith("File "):
                        file_info.append(lines[k].strip())

                if file_info:
                    error_info["file_traceback"] = file_info
                break

        return error_info

    def _parse_error_exception_line(self, log_output: str) -> dict:
        """Parse direct error/exception lines."""
        error_info = {}

        # Look for lines ending with Error: or Exception:
        error_lines = [
            line for line in log_output.splitlines() if re.search(r"\w+(Error|Exception):", line)
        ]

        if error_lines:
            last_error = error_lines[-1].strip()
            if ":" in last_error:
                error_parts = last_error.split(":", 1)
                error_info["error_type"] = error_parts[0].strip()
                error_info["error_message"] = error_parts[1].strip()
            else:
                error_info["error_message"] = last_error

        return error_info

    def _parse_numpy_error(self, log_output: str) -> dict:
        """Parse numpy/scientific computing specific errors."""
        error_info = {"error_category": "scientific_computing"}

        # Specific numpy error patterns
        numpy_patterns = {
            r"minimum\(\) takes from (\d+) to (\d+) positional arguments but (\d+) were given": {
                "error_type": "TypeError",
                "error_message": "np.minimum() called with incorrect number of arguments",
                "suggested_fix": "Use np.minimum.reduce() for multiple arrays or chain np.minimum() calls",
            },
            r"maximum\(\) takes from (\d+) to (\d+) positional arguments but (\d+) were given": {
                "error_type": "TypeError",
                "error_message": "np.maximum() called with incorrect number of arguments",
                "suggested_fix": "Use np.maximum.reduce() for multiple arrays or chain np.maximum() calls",
            },
            r"could not broadcast input array": {
                "error_type": "ValueError",
                "error_message": "Array broadcasting error",
                "suggested_fix": "Check array shapes for compatibility",
            },
            r"division by zero": {
                "error_type": "RuntimeWarning",
                "error_message": "Division by zero encountered",
                "suggested_fix": "Add checks for zero values before division",
            },
        }

        for pattern, error_details in numpy_patterns.items():
            if re.search(pattern, log_output):
                error_info.update(error_details)
                break

        return error_info

    def _parse_ray_error(self, log_output: str) -> dict:
        """Parse Ray-specific errors."""
        error_info = {"error_category": "ray_distributed"}

        ray_patterns = {
            "WorkerCrashedError": "Ray worker process crashed during execution",
            "RayTaskError": "Ray task execution failed",
            "ray.exceptions": "Ray framework error occurred",
        }

        for pattern, description in ray_patterns.items():
            if pattern in log_output:
                error_info["error_type"] = pattern
                error_info["error_message"] = description
                break

        return error_info

    def _parse_generic_error(self, log_output: str, pattern: str) -> dict:
        """Parse generic errors not covered by specific parsers."""
        error_info = {"error_category": "generic"}

        # Try to find context around the matched pattern
        lines = log_output.splitlines()
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                error_info["error_line"] = line.strip()
                error_info["error_message"] = f"Error detected with pattern: {pattern}"

                # Include surrounding context
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                error_info["surrounding_context"] = "\n".join(lines[context_start:context_end])
                break

        return error_info

    def _extract_error_context(self, log_output: str, context_lines: int = 10) -> str:
        """Extract relevant context around error locations."""
        lines = log_output.splitlines()

        # Find lines with error indicators
        error_line_indices = []
        error_indicators = ["Error:", "Exception:", "Traceback", "FAILED", "CRITICAL:"]

        for i, line in enumerate(lines):
            if any(indicator in line for indicator in error_indicators):
                error_line_indices.append(i)

        if not error_line_indices:
            # No specific error lines found, return last few lines
            return "\n".join(lines[-context_lines:])

        # Extract context around the last error
        last_error_idx = error_line_indices[-1]
        start_idx = max(0, last_error_idx - context_lines // 2)
        end_idx = min(len(lines), last_error_idx + context_lines // 2)

        return "\n".join(lines[start_idx:end_idx])

    def _extract_error_info(self, log_output: str) -> dict:
        """
        Legacy method for backward compatibility.
        Extract error information from failed job logs.

        Args:
            log_output: Raw log output containing error traces

        Returns:
            Dictionary containing error details
        """
        # Use simple error detection
        if self._has_execution_errors(log_output):
            return {"error_info": log_output}
        else:
            return {}

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
        """
        Generic parser for artifacts dict that can handle dynamic numpy arrays and other data types.

        Args:
            artifacts_str: String representation of artifacts dictionary

        Returns:
            Parsed artifacts dictionary
        """
        artifacts = {}

        # Parse numpy arrays first (most complex)
        artifacts.update(self._extract_numpy_arrays(artifacts_str))

        # Parse simple key-value pairs (skip already processed keys)
        artifacts.update(self._extract_simple_key_values(artifacts_str, set(artifacts.keys())))

        return artifacts

    def _extract_numpy_arrays(self, text: str) -> dict:
        """Extract all numpy arrays from the artifacts string."""
        arrays = {}
        array_pattern = re.compile(r"'([^']+)':\s*array\(\[(.*?)\]\)", re.DOTALL)

        for match in array_pattern.finditer(text):
            key = match.group(1)
            array_content = match.group(2)

            try:
                arrays[key] = self._parse_numpy_array_content(array_content)
            except Exception as e:
                logger.warning(f"Failed to parse array for key '{key}': {e}")

        return arrays

    def _extract_simple_key_values(self, text: str, skip_keys: set) -> dict:
        """Extract simple key-value pairs, skipping already processed keys."""
        values = {}
        simple_pattern = re.compile(r"'([^']+)':\s*([^,}\n]+)")

        for match in simple_pattern.finditer(text):
            key = match.group(1)

            # Skip if this key was already processed
            if key in skip_keys:
                continue

            value_str = match.group(2).strip()
            values[key] = self._parse_simple_value(key, value_str)

        return values

    def _parse_simple_value(self, key: str, value_str: str):
        """Parse a simple value (string, number, boolean)."""
        try:
            if value_str.startswith("'") and value_str.endswith("'"):
                return value_str[1:-1]  # String value
            elif value_str.lower() in ["true", "false"]:
                return value_str.lower() == "true"  # Boolean value
            elif "." in value_str:
                return float(value_str)  # Float value
            elif value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
                return int(value_str)  # Integer value
            else:
                return value_str  # Fallback to string
        except Exception as e:
            logger.warning(f"Failed to parse value for key '{key}': {e}")
            return value_str  # Fallback to string

    def _parse_numpy_array_content(self, array_content: str) -> list:
        """
        Parse the content of a numpy array (inside the brackets).
        Handles both 1D and 2D arrays.

        Args:
            array_content: String content inside array brackets

        Returns:
            List of numeric values
        """
        content = array_content.strip()

        if self._is_2d_array(content):
            return self._parse_2d_array(content)
        else:
            return self._parse_1d_array(content)

    def _is_2d_array(self, content: str) -> bool:
        """Check if content represents a 2D array."""
        return "[" in content and "]" in content

    def _parse_2d_array(self, content: str) -> list:
        """Parse 2D array content."""
        values = []
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        for line in lines:
            line_values = self._extract_numbers_from_line(line)
            values.extend(line_values)

        return values

    def _parse_1d_array(self, content: str) -> list:
        """Parse 1D array content."""
        return self._extract_numbers_from_line(content)

    def _extract_numbers_from_line(self, line: str) -> list:
        """Extract numeric values from a single line."""
        # Remove brackets and commas, split by whitespace
        line_clean = re.sub(r"[\[\],]", " ", line)
        return [float(x) for x in line_clean.split() if x and self._is_numeric_string(x)]

    def _is_numeric_string(self, s: str) -> bool:
        """Check if string represents a valid number."""
        return s.replace(".", "").replace("-", "").isdigit()

    def _extract_simple_metrics(self, metrics_str: str) -> dict:
        """
        Extract metrics from a string representation of a dictionary.
        Handles various types including complex structures dynamically.

        Args:
            metrics_str: String representation of metrics dictionary

        Returns:
            Parsed metrics dictionary
        """
        metrics = {}

        # First try to parse the entire string as a dictionary
        try:
            # Clean up the string and try ast.literal_eval
            cleaned_str = self._preprocess_log_line(metrics_str)
            parsed_dict = ast.literal_eval(cleaned_str)
            if isinstance(parsed_dict, dict):
                return parsed_dict
        except Exception:
            pass

        # Fallback: Extract individual key-value pairs with enhanced patterns
        patterns = [
            # Simple patterns
            (r"'([^']+)':\s*([\d.-]+)", self._parse_numeric_value),  # 'key': number
            (r'"([^"]+)":\s*([\d.-]+)', self._parse_numeric_value),  # "key": number
            (r"'([^']+)':\s*(True|False|true|false)", self._parse_boolean_value),  # 'key': boolean
            (r'"([^"]+)":\s*(True|False|true|false)', self._parse_boolean_value),  # "key": boolean
            # Complex patterns for tuples, lists, dicts
            (
                r"'([^']+)':\s*(\([^}]+\}?\))",
                self._parse_complex_metric_value,
            ),  # 'key': (tuple_content)
            (
                r"'([^']+)':\s*(\[[^\]]*\])",
                self._parse_complex_metric_value,
            ),  # 'key': [list_content]
            (
                r"'([^']+)':\s*(\{[^}]*\})",
                self._parse_complex_metric_value,
            ),  # 'key': {dict_content}
        ]

        for pattern_str, parser in patterns:
            pattern = re.compile(pattern_str, re.DOTALL)
            for match in pattern.finditer(metrics_str):
                key = match.group(1)
                value_str = match.group(2)

                # Skip if key already processed
                if key in metrics:
                    continue

                try:
                    metrics[key] = parser(value_str)
                except Exception as e:
                    logger.warning(f"Failed to parse metric value for key '{key}': {e}")
                    metrics[key] = value_str  # Fallback to string

        return metrics

    def _parse_numeric_value(self, value_str: str):
        """Parse numeric value (int or float)."""
        if "." in value_str:
            return float(value_str)
        elif value_str.lstrip("-").isdigit():
            return int(value_str)
        else:
            return value_str

    def _parse_boolean_value(self, value_str: str):
        """Parse boolean value."""
        return value_str.lower() == "true"

    def _parse_complex_metric_value(self, value_str: str):
        """
        Parse complex metric values like tuples, lists, or dictionaries.

        Args:
            value_str: String representation of a complex value

        Returns:
            Parsed Python object or original string if parsing fails
        """
        try:
            # First try ast.literal_eval for safe evaluation of literals
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, try manual parsing for common patterns
            try:
                return self._parse_tuple_or_dict_manually(value_str)
            except Exception:
                # If all parsing fails, return as string
                return value_str

    def _parse_tuple_or_dict_manually(self, value_str: str):
        """
        Manually parse tuple or dict structures that ast.literal_eval might reject.
        """
        value_str = value_str.strip()

        # Handle tuple format: (value1, value2, ...)
        if value_str.startswith("(") and value_str.endswith(")"):
            # Extract tuple content
            content = value_str[1:-1].strip()

            # Split by comma at the top level (not inside nested structures)
            elements = self._split_by_comma_top_level(content)
            parsed_elements = []

            for element in elements:
                element = element.strip()
                try:
                    # Try to parse each element
                    if element.startswith("{") and element.endswith("}"):
                        # Dictionary element
                        parsed_elements.append(self._parse_dict_content(element))
                    elif element.lower() in ["true", "false"]:
                        # Boolean element
                        parsed_elements.append(element.lower() == "true")
                    elif self._is_numeric_string(element.replace(".", "").replace("-", "")):
                        # Numeric element
                        parsed_elements.append(float(element) if "." in element else int(element))
                    else:
                        # String element (remove quotes if present)
                        if element.startswith("'") and element.endswith("'"):
                            parsed_elements.append(element[1:-1])
                        elif element.startswith('"') and element.endswith('"'):
                            parsed_elements.append(element[1:-1])
                        else:
                            parsed_elements.append(element)
                except Exception:
                    # If individual element parsing fails, keep as string
                    parsed_elements.append(element)

            return tuple(parsed_elements)

        # Handle dict format: {...}
        elif value_str.startswith("{") and value_str.endswith("}"):
            return self._parse_dict_content(value_str)

        # Handle list format: [...]
        elif value_str.startswith("[") and value_str.endswith("]"):
            content = value_str[1:-1].strip()
            elements = self._split_by_comma_top_level(content)
            return [self._parse_simple_element(elem.strip()) for elem in elements]

        else:
            return value_str

    def _split_by_comma_top_level(self, content: str) -> list:
        """Split content by commas, but only at the top level (not inside nested structures)."""
        elements = []
        current_element = ""
        bracket_depth = 0
        brace_depth = 0
        paren_depth = 0
        in_quotes = False
        quote_char = None

        for char in content:
            if not in_quotes:
                if char in ['"', "'"]:
                    in_quotes = True
                    quote_char = char
                elif char == "[":
                    bracket_depth += 1
                elif char == "]":
                    bracket_depth -= 1
                elif char == "{":
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                elif char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                elif char == "," and bracket_depth == 0 and brace_depth == 0 and paren_depth == 0:
                    elements.append(current_element)
                    current_element = ""
                    continue
            else:
                if char == quote_char:
                    in_quotes = False
                    quote_char = None

            current_element += char

        if current_element:
            elements.append(current_element)

        return elements

    def _parse_dict_content(self, dict_str: str) -> dict:
        """Parse dictionary content manually."""
        try:
            # Try ast.literal_eval first
            return ast.literal_eval(dict_str)
        except Exception:
            # Manual parsing for simpler cases
            result = {}
            content = dict_str[1:-1].strip()  # Remove { }

            # Simple regex for key-value pairs
            pairs = re.findall(r"'([^']+)':\s*([^,}]+)", content)
            for key, value in pairs:
                value = value.strip()
                result[key] = self._parse_simple_element(value)

            return result

    def _parse_simple_element(self, element: str):
        """Parse a simple element (number, boolean, string)."""
        element = element.strip()

        if element.lower() in ["true", "false"]:
            return element.lower() == "true"
        elif element.startswith("'") and element.endswith("'"):
            return element[1:-1]
        elif element.startswith('"') and element.endswith('"'):
            return element[1:-1]
        elif element.startswith("[") and element.endswith("]"):
            # Simple list
            list_content = element[1:-1].strip()
            if not list_content:
                return []
            items = [item.strip() for item in list_content.split(",")]
            return [self._parse_simple_element(item) for item in items]
        elif self._is_numeric_string(element.replace(".", "").replace("-", "")):
            return float(element) if "." in element else int(element)
        else:
            return element

    def _extract_simple_artifacts(self, artifacts_str: str) -> dict:
        """
        Fallback method to extract simple key-value pairs when full parsing fails.

        Args:
            artifacts_str: String representation of artifacts

        Returns:
            Dictionary of simple key-value pairs
        """
        artifacts = {}

        # Look for simple patterns like 'key': value
        simple_pairs = re.findall(r"'([^']+)':\s*([^,}\n]+)", artifacts_str)

        for key, value in simple_pairs:
            value = value.strip()
            artifacts[key] = self._parse_fallback_value(value)

        return artifacts

    def _parse_fallback_value(self, value: str):
        """Parse a value with simple fallback logic."""
        try:
            if value.startswith("'") and value.endswith("'"):
                return value[1:-1]  # String value
            elif value.lower() in ["true", "false"]:
                return value.lower() == "true"  # Boolean value
            elif "." in value:
                return float(value)  # Float value
            elif value.isdigit():
                return int(value)  # Integer value
            else:
                return value  # Fallback to string
        except:
            return value  # Ultimate fallback
