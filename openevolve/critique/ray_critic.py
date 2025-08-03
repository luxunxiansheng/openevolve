# we will use ray job client to submit the evaluation job 
import re
import time
import logging

from ray.job_submission import JobSubmissionClient, JobStatus

from openevolve.evaluation.critic import EvaluationResult, Critic

logger = logging.getLogger(__name__)


class RayPythonCritic(Critic):
    """ Controller for evaluating Python programs using Ray Job Submission Client.
    This class handles the submission of evaluation jobs to a Ray cluster and monitors their status.
    It also extracts evaluation results from the job logs.
    """ 
    def __init__(self,
                 ray_cluster_head_ip:str="http//:localhost:8265",
                 ) -> None:

        self.job_client = JobSubmissionClient(ray_cluster_head_ip)
        


    def evaluate(self, **kwargs) -> EvaluationResult:
        """ 
           evaluate and log the metrics and artifacts of the given program code.
        """

        python_file_path = kwargs.get("python_file_path")
        program_id = kwargs.get("program_id")
        runtime_env = kwargs.get("runtime_env", {}) 

        if not python_file_path:
            raise ValueError("python_file_path must be provided for evaluation.")
        elif not python_file_path.endswith(".py"):
            raise ValueError("python_file_path must point to a valid Python file.")
        
        if not program_id:
            raise ValueError("program_id must be provided for evaluation.")
        
        if not runtime_env:
            logger.warning("No runtime environment provided, using default settings.")
        
        logger.info(f"Submitting evaluation job with Python file: {python_file_path} and program ID: {program_id} with runtime environment: {runtime_env}")
       
        # Check if job with same ID already exists and handle it
        self._handle_existing_job(program_id)

        entrypoint="python " + str(python_file_path)

        submission_id = self.job_client.submit_job(
            entrypoint=entrypoint,
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

    def _handle_existing_job(self, job_id: str) -> None:
        """
        Check if a job with the given ID already exists and handle it appropriately.
        Cancel/stop running jobs, or delete completed ones.
        """
        try:
            # Check if job exists
            existing_status = self.job_client.get_job_status(job_id)
            logger.info(f"Found existing job {job_id} with status: {existing_status}")
            
            if existing_status in [JobStatus.PENDING, JobStatus.RUNNING]:
                logger.info(f"Stopping existing job {job_id} before submitting new one.")
                self.job_client.stop_job(job_id)
                
                # Wait for job to stop
                max_wait_time = 30  # seconds
                wait_time = 0
                while wait_time < max_wait_time:
                    status = self.job_client.get_job_status(job_id)
                    if status in [JobStatus.STOPPED, JobStatus.FAILED, JobStatus.SUCCEEDED]:
                        logger.info(f"Job {job_id} stopped with status: {status}")
                        break
                    time.sleep(1)
                    wait_time += 1
                else:
                    logger.warning(f"Job {job_id} did not stop within {max_wait_time} seconds")
            
            # Delete the job regardless of its final status to allow reuse of the submission_id
            logger.info(f"Deleting existing job {job_id} to reuse submission_id")
            self.job_client.delete_job(job_id)
            
            # Wait a moment for deletion to complete
            time.sleep(1)
                
        except Exception as e:
            # Job might not exist, which is fine for new submissions
            logger.debug(f"No existing job found with ID {job_id}: {e}")





