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
        print(f"函数 '{func.__name__}' 执行时间: {execution_time:.4f} 秒")

        return result

    return wrapper