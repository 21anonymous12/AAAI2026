import multiprocessing
import time

def check_timeout_column_name(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  

    if process.is_alive():  
        process.terminate()
        process.join()  
        print(f"Function timed out after {timeout} seconds.")
        return Exception("Function execution exceeded the timeout limit.")
    else:
        return queue.get() 


def check_timeout_PoT(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  

    if process.is_alive():  
        process.terminate()
        process.join()  
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  


def check_timeout_CoT(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  

    if process.is_alive():  
        process.terminate()
        process.join()  
        print(f"Function timed out after {timeout} seconds.")
        return {"solution": "inference timeout error", 'answer' : "inference timeout error"}
    else:
        return queue.get()  


def check_timeout_PoT_refine(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  

    if process.is_alive():  
        process.terminate()
        process.join()  
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  


def check_timeout_CoT_refine(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  

    if process.is_alive():  
        process.terminate()
        process.join() 
        print(f"Function timed out after {timeout} seconds.")
        return {"solution": "inference timeout error", 'answer' : "inference timeout error"}
    else:
        return queue.get() 


def check_timeout_text2sql(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  

    if process.is_alive():  
        process.terminate()
        process.join()  
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  


def check_timeout_text2sql_refine(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  

    if process.is_alive():  
        process.terminate()
        process.join()  
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  
