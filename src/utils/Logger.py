import os
from datetime import datetime

class Logger:
    _instances = {}
    _instance_lock = {}

    def __new__(cls, log_dir='tests', log_file='log', rank=0, rep="all"):
        key = (rank, rep)

        # Ensure thread-safe access to _instances
        if key not in cls._instances:
            # Lock per key to prevent race conditions between processes
            cls._instance_lock[key] = cls._instance_lock.get(key, False)
            if not cls._instance_lock[key]:
                cls._instance_lock[key] = True
                try:
                    # Create the directory if it doesn't exist
                    os.makedirs(log_dir, exist_ok=True)

                    # Generate the current timestamp with microseconds
                    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

                    # Concatenate the log_file name with the current timestamp, rank, and rep
                    log_file_with_rank_rep = f'{log_file}_{current_time}_rank{rank}_rep{rep}.log'

                    # Combine directory and file name
                    log_file_path = os.path.join(log_dir, log_file_with_rank_rep)

                    # Create the instance
                    instance = super(Logger, cls).__new__(cls)
                    instance.log_file = log_file_path
                    instance._initialize_log()

                    cls._instances[key] = instance
                finally:
                    cls._instance_lock[key] = False

        return cls._instances[key]

    def _initialize_log(self):
        with open(self.log_file, 'a') as file:
            file.write("-----------------------------------------------\n")
            file.write(f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")
            file.write("-----------------------------------------------\n")

    def log(self, message: str):
        with open(self.log_file, 'a') as file:
            file.write(f"### - {message}\n")
