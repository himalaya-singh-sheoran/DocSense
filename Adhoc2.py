import threading
import time
import heapq


class JobExecutor:
    def __init__(self, max_threads=5):
        self.job_queue = []  # Priority queue for jobs
        self.lock = threading.Lock()  # Lock for thread-safe access
        self.max_threads = max_threads  # Maximum concurrent threads
        self.active_threads = 0  # Count of active threads
        self.executor_thread = threading.Thread(target=self._job_worker, daemon=True)
        self.running = True  # To control the executor's lifecycle

    def start(self):
        """Start the job executor thread."""
        self.executor_thread.start()

    def stop(self):
        """Stop the job executor thread."""
        self.running = False
        self.executor_thread.join()

    def add_job(self, job_id, priority, job_func, *args, **kwargs):
        """
        Add a job to the priority queue.
        :param job_id: Unique identifier for the job
        :param priority: Priority of the job (lower value = higher priority)
        :param job_func: Function to execute as the job
        :param args: Positional arguments for the job function
        :param kwargs: Keyword arguments for the job function
        """
        with self.lock:
            heapq.heappush(self.job_queue, (priority, job_id, job_func, args, kwargs))
            print(f"Job {job_id} added with priority {priority}.")

    def _job_worker(self):
        """
        Continuously monitor and execute jobs from the queue.
        """
        while self.running:
            with self.lock:
                if self.active_threads < self.max_threads and self.job_queue:
                    # Fetch the highest priority job
                    priority, job_id, job_func, args, kwargs = heapq.heappop(self.job_queue)
                    print(f"Starting job {job_id} with priority {priority}.")
                    self.active_threads += 1

                    # Start a thread to execute the job
                    threading.Thread(target=self._execute_job, args=(job_id, job_func, *args), kwargs=kwargs).start()

            time.sleep(0.5)  # Short sleep to avoid busy waiting

    def _execute_job(self, job_id, job_func, *args, **kwargs):
        """
        Wrapper to execute a job and handle thread completion.
        """
        try:
            job_func(*args, **kwargs)
            print(f"Job {job_id} completed.")
        except Exception as e:
            print(f"Error executing job {job_id}: {e}")
        finally:
            with self.lock:
                self.active_threads -= 1
                
