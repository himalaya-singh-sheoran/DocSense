import heapq
from datetime import datetime, timedelta

class Job:
    def __init__(self, id, start_time, min_time, avg_time, max_time):
        self.id = id
        self.start_time = start_time
        self.min_time = min_time
        self.avg_time = avg_time
        self.max_time = max_time
        self.remaining_time = max_time

    def __lt__(self, other):
        return self.remaining_time < other.remaining_time

def optimized_schedule_jobs(jobs, start_of_day, end_of_day):
    # Convert string times to datetime objects
    for job in jobs:
        job.start_time = datetime.strptime(job.start_time, "%H:%M")
    
    # Initialize the priority queue
    job_queue = []
    heapq.heapify(job_queue)
    
    current_time = start_of_day
    scheduled_jobs = []
    
    while current_time < end_of_day:
        # Add jobs that are ready to start to the priority queue
        for job in jobs:
            if job.start_time == current_time:
                heapq.heappush(job_queue, job)
        
        # Dynamic prioritization: prioritize jobs that can be completed within the remaining time
        if job_queue:
            # Sort jobs in the queue based on their remaining time and urgency
            job_queue.sort(key=lambda job: (job.remaining_time, (end_of_day - current_time).total_seconds() / 60 - job.remaining_time))
            
            # Execute the job with the highest priority
            current_job = heapq.heappop(job_queue)
            if current_time + timedelta(minutes=current_job.remaining_time) <= end_of_day:
                # Job can be completed today
                current_time += timedelta(minutes=current_job.remaining_time)
                scheduled_jobs.append(current_job.id)
            else:
                # Job can't be completed today, update remaining time and reschedule
                current_job.remaining_time -= (end_of_day - current_time).total_seconds() / 60
                current_time = end_of_day
                heapq.heappush(job_queue, current_job)
        else:
            # No jobs ready to run, move to the next minute
            current_time += timedelta(minutes=1)
    
    return scheduled_jobs

# Example usage
jobs = [
    Job(1, "09:00", 10, 15, 20),
    Job(2, "10:00", 5, 10, 15),
    Job(3, "11:00", 20, 25, 30),
    Job(4, "09:30", 10, 12, 15),
    Job(5, "10:15", 5, 7, 10)
]

start_of_day = datetime.strptime("09:00", "%H:%M")
end_of_day = datetime.strptime("17:00", "%H:%M")

scheduled_jobs = optimized_schedule_jobs(jobs, start_of_day, end_of_day)
print("Scheduled Jobs:", scheduled_jobs)


