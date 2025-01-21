import threading
import time
import heapq

class JobExecutor:
    def __init__(self, max_threads=5):
        self.job_queue = []  # Priority queue for jobs
        self.lock = threading.Lock()  # Lock for thread-safe access
        self.max_threads = max_threads  # Maximum concurrent threads
        self.active_threads = 0  # Count of active threads

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

    def execute_jobs(self):
        """
        Execute jobs based on their priority. At most, `max_threads` jobs are executed concurrently.
        """
        while True:
            with self.lock:
                if self.active_threads < self.max_threads and self.job_queue:
                    # Fetch the highest priority job
                    priority, job_id, job_func, args, kwargs = heapq.heappop(self.job_queue)
                    print(f"Starting job {job_id} with priority {priority}.")
                    self.active_threads += 1

                    # Start a thread to execute the job
                    threading.Thread(target=self._execute_job, args=(job_id, job_func, *args), kwargs=kwargs).start()
                elif not self.job_queue and self.active_threads == 0:
                    # Exit when all jobs are completed
                    print("All jobs completed.")
                    break

            time.sleep(1)  # Prevent busy waiting

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


# Example usage
if __name__ == "__main__":
    # Define a sample job function
    def sample_job(job_id, duration):
        print(f"Executing job {job_id} for {duration} seconds.")
        time.sleep(duration)

    # Initialize the JobExecutor
    executor = JobExecutor(max_threads=5)

    # Add jobs with varying priorities and durations
    executor.add_job("job1", priority=3, job_func=sample_job, job_id="job1", duration=5)
    executor.add_job("job2", priority=1, job_func=sample_job, job_id="job2", duration=3)
    executor.add_job("job3", priority=2, job_func=sample_job, job_id="job3", duration=4)
    executor.add_job("job4", priority=5, job_func=sample_job, job_id="job4", duration=2)
    executor.add_job("job5", priority=4, job_func=sample_job, job_id="job5", duration=6)
    executor.add_job("job6", priority=0, job_func=sample_job, job_id="job6", duration=1)

    # Start executing jobs
    executor.execute_jobs()
