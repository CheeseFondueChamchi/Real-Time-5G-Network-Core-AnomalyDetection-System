import time
from datetime import datetime, timedelta

def timer(func) :

    def wrapper(*args, **kwargs) :

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        td = timedelta(seconds=elapsed_time)
        
        print(f"{func.__name__} 함수의 실행 시간 : {str(td)}초")

        return result

    return wrapper
