from dataclasses import dataclass



@dataclass
class CriticConfig:
    # Constants for job management
    default_ray_head_ip = "http://127.0.0.1:8265"
    job_timeout_seconds = 3600  # 1 hour
    status_check_interval = 5  # seconds
    job_stop_wait_time = 30  # seconds
    deletion_wait_time = 1  # second

    

