import heapq
from datetime import datetime, timedelta
from collections import defaultdict, deque

class Job:
    def __init__(self, id, start_time, min_time, avg_time, max_time, prerequisites=[]):
        self.id = id
        self.start_time = start_time
        self.min_time = min_time
        self.avg_time = avg_time
        self.max_time = max_time
        self.remaining_time = max_time
        self.prerequisites = prerequisites

    def __lt__(self, other):
        return self.remaining_time < other.remaining_time

def topological_sort(jobs):
    # Create a graph and a degree of incoming edges dictionary
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for job in jobs:
        in_degree[job.id] = 0
    
    for job in jobs:
        for prerequisite in job.prerequisites:
            graph[prerequisite].append(job.id)
            in_degree[job.id] += 1
    
    # Queue for nodes with no incoming edges
    queue = deque([job.id for job in jobs if in_degree[job.id] == 0])
    sorted_jobs = []
    
    while queue:
        node = queue.popleft()
        sorted_jobs.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles in the graph
    if len(sorted_jobs) != len(jobs):
        raise Exception("There exists a cycle in the job dependencies")
    
    return sorted_jobs

def optimized_schedule_jobs(jobs, start_of_day, end_of_day):
    # Convert string times to datetime objects
    job_map = {}
    for job in jobs:
        job.start_time = datetime.strptime(job.start_time, "%H:%M")
        job_map[job.id] = job
    
    # Perform topological sort
    sorted_job_ids = topological_sort(jobs)
    
    # Initialize the priority queue
    job_queue = []
    heapq.heapify(job_queue)
    
    current_time = start_of_day
    scheduled_jobs = []
    job_ready = set()
    
    while current_time < end_of_day:
        # Add jobs that are ready to start and have their prerequisites met to the priority queue
        for job_id in sorted_job_ids:
            job = job_map[job_id]
            if job.start_time <= current_time and all(prerequisite in scheduled_jobs for prerequisite in job.prerequisites):
                if job.id not in job_ready:
                    heapq.heappush(job_queue, job)
                    job_ready.add(job.id)
        
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
    Job(2, "10:00", 5, 10, 15, prerequisites=[1]),
    Job(3, "11:00", 20, 25, 30),
    Job(4, "09:30", 10, 12, 15, prerequisites=[1]),
    Job(5, "10:15", 5, 7, 10, prerequisites=[2, 4])
]

start_of_day = datetime.strptime("09:00", "%H:%M")
end_of_day = datetime.strptime("17:00", "%H:%M")

scheduled_jobs = optimized_schedule_jobs(jobs, start_of_day, end_of_day)
print("Scheduled Jobs:", scheduled_jobs)
