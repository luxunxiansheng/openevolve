# we will use ray job client to submit the evaluation job 

import time
import logging

from ray.job_submission import JobSubmissionClient, JobStatus

from openevolve.evaluation.evaluator import EvaluationResult

logger = logging.getLogger(__name__)

class RayEvaluationController:
    """ Controller for evaluating Python programs using Ray Job Submission Client.
    This class handles the submission of evaluation jobs to a Ray cluster and monitors their status.
    It also extracts evaluation results from the job logs.
    """ 

    def __init__(self,
                 ray_cluster_head_ip:str="http//:localhost:8265",
                 ) -> None:

        self.job_client = JobSubmissionClient(ray_cluster_head_ip)

    def evaluate_python(self, 
                        python_file_path: str,
                        program_id: str,
                        runtime_env: dict)-> EvaluationResult:
        
        """        Submit a Python evaluation job to the Ray cluster.
        Args:
            python_file_path (str): Path to the Python file to be executed.
            runtime_env (dict): Runtime environment configuration for the job.
        Returns:
            str: Job ID of the submitted job.
        """ 

        logger.info(f"Submitting evaluation job with Python file: {python_file_path}")

        submission_id = self.job_client.submit_job(
            entrypoint="python " + python_file_path,
            runtime_env=runtime_env,
            submission_id=program_id,  # Use program_id as the job ID
            
        )    

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

            elapsed_time = time.time() - start_time
            if elapsed_time > 3600:  # Timeout after 1 hour
                logger.error(f"Job {submission_id} timed out after 1 hour.")
                break           

            time.sleep(5)  # Check the job status every 5 seconds

        # Analyze the job's log to extract the evaluation result
        log_output = self.job_client.get_job_logs(submission_id)
        eval_result = self._extract_evaluation_result_from_logs(log_output)
        return eval_result

    def _extract_evaluation_result_from_logs(self, log_output: str) -> EvaluationResult:
        """
        Extract EvaluationResult from evaluator's default log output.
        """
        import re
        metrics = {}
        artifacts = {}

        metric_re = re.compile(r"^Metric ([^:]+): (.+)$")
        artifact_str_re = re.compile(r"^Artifact ([^:]+): (.+)$")
        artifact_bytes_re = re.compile(r"^Artifact ([^:]+): (\d+) bytes$")

        for line in log_output.splitlines():
            m = metric_re.match(line)
            if m:
                key, value = m.group(1), m.group(2)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass
                continue

            m = artifact_bytes_re.match(line)
            if m:
                key, size = m.group(1), int(m.group(2))
                artifacts[key] = b"\x00" * size
                continue

            m = artifact_str_re.match(line)
            if m:
                key, value = m.group(1), m.group(2)
                artifacts[key] = value
                continue

        return EvaluationResult(metrics=metrics, artifacts=artifacts)





