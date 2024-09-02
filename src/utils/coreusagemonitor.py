import psutil
import time
from typing import Callable, Any

class CoreUsageMonitor:
    """
    A class to monitor CPU core usage during the execution of functions.
    Designed to work with Joblib on a single node.
    """

    def __init__(self):
        """
        Initializes the CoreUsageMonitor.
        """
        self.num_cores = psutil.cpu_count(logical=False)
        print(f"Available cores: {self.num_cores}")

    def monitor(self, func: Callable) -> Callable:
        """
        A decorator to monitor the number of CPU cores used during the execution of a function.

        Args:
            func: The function to be monitored.

        Returns:
            A wrapper function that monitors core usage during the function execution.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Record the CPU times for each core before execution
            start_per_core_times = psutil.cpu_times_percent(percpu=True)

            result = func(*args, **kwargs)

            # Record the CPU times for each core after execution
            end_per_core_times = psutil.cpu_times_percent(percpu=True)

            # Calculate CPU usage per core during the function execution
            core_usage = []
            for start, end in zip(start_per_core_times, end_per_core_times):
                usage = {
                    "user": end.user - start.user,
                    "system": end.system - start.system,
                    "idle": end.idle - start.idle,
                }
                core_usage.append(usage)

            # Count cores that were active (not idle) during execution
            active_cores = sum(1 for usage in core_usage if usage["idle"] < (usage["user"] + usage["system"]))

            print(f"{func.__name__} used {active_cores} cores during execution.")

            return result

        return wrapper
