import numpy as np
from datetime import datetime, timezone, timedelta

class TimeBenchmark:
    def __init__(self):
        self.times: np.ndarray[timedelta] = np.empty(8, dtype=timedelta)
        self.size: int = 0
        self.capacity: int = 8
        self.begin_time: datetime = datetime.n
    def begin_benchmark(self) -> None:
        self.begin_time = datetime.now(timezone.utc)
    def end_benchmark(self) -> None:
        time_passed: timedelta = datetime.now(timezone.utc) - self.begin_time
        if self.size >= self.capacity:
            self.capacity *= 2
            self.times = np.resize(self.times, self.capacity)
        self.times[self.size] = time_passed
        self.size += 1
    def get_average_time(self) -> float:
        return np.average(self.times)
