import logging
import time
import functools
from typing import Callable, Any

def timing(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        logging.info(f"function '{func.__name__}' cost: {execution_time:.4f}s")

        return result

    return wrapper