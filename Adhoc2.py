import threading
import time
import heapq
import json
from confluent_kafka import Consumer

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
TOPIC = "job_topic"
GROUP_ID = "job_executor_group"

# Define the JobExecutor class
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


# Define a sample job function
def sample_job(job_id, duration):
    print(f"Executing job {job_id} for {duration} seconds.")
    time.sleep(duration)


# Kafka consumer thread
def kafka_consumer_thread(executor):
    """
    Kafka consumer thread that consumes messages and adds jobs to the JobExecutor.
    :param executor: Instance of JobExecutor to add jobs to.
    """
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe([TOPIC])
    print("Kafka consumer started and subscribed to topic.")

    try:
        while True:
            msg = consumer.poll(1.0)  # Poll for messages with a timeout of 1 second
            if msg is None:
                continue  # No message received
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # Parse the Kafka message
            try:
                job_data = msg.value().decode('utf-8')  # Decode the message
                print(f"Consumed message: {job_data}")

                # Example: Assume job_data is a JSON string with job_id, priority, and duration
                job = json.loads(job_data)

                # Add the job to the executor
                executor.add_job(
                    job_id=job['job_id'],
                    priority=job['priority'],
                    job_func=sample_job,
                    job_id=job['job_id'],
                    duration=job['duration']
                )
            except Exception as e:
                print(f"Error processing message: {e}")
    except Exception as e:
        print(f"Kafka consumer thread error: {e}")
    finally:
        consumer.close()
        print("Kafka consumer closed.")


# Main application
if __name__ == "__main__":
    # Initialize the JobExecutor
    executor = JobExecutor(max_threads=5)

    # Start the JobExecutor
    executor.start()

    # Start the Kafka consumer thread
    consumer_thread = threading.Thread(target=kafka_consumer_thread, args=(executor,), daemon=True)
    consumer_thread.start()

    # Let the application run indefinitely
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        executor.stop()
        consumer_thread.join()
