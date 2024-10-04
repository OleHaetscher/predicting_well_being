import time
from datetime import datetime
from typing import Callable, Any
from src.utils.Logger import Logger

class Timer:
    """
    A utility class for timing the execution of methods. It can be used as a decorator
    to time class methods and pass the timing information to a logger.
    """

    def __init__(self, logger: Logger):
        """
        Initializes the Timer with a logger instance.

        Args:
            logger: An instance of a logger that has a `log` method for logging messages.
        """
        self.logger = logger

    def _decorator(self, func: Callable) -> Callable:
        """
        A method to decorate functions to log their execution time.

        Args:
            func: The function to be decorated.

        Returns:
            Callable: The wrapped function with timing logic.
        """
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            hours, remainder = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_statement = f"{func.__name__} executed in {int(hours):02}h {int(minutes):02}m {seconds:.2f}s"

            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{time_statement} at {current_datetime}"

            self.logger.log("-----------------------")
            self.logger.log(log_message)
            self.logger.log("-----------------------")
            return result

        return wrapper
