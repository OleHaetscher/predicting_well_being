import os
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, log_dir='tests', log_file='log'):
        if cls._instance is None:
            # Create the directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Generate the current timestamp
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            # Concatenate the provided log_file name with the current timestamp
            log_file = f'{log_file}_{current_time}.log'

            # Combine directory and file name
            log_file_path = os.path.join(log_dir, log_file)

            # Create the instance
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.log_file = log_file_path
            cls._instance._initialize_log()

        return cls._instance

    def _initialize_log(self):
        with open(self.log_file, 'a') as file:
            file.write("-----------------------------------------------\n")
            file.write(f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")
            file.write("-----------------------------------------------\n")

    def log(self, message: str):
        with open(self.log_file, 'a') as file:
            file.write(f"### - {message}\n")
