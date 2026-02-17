"""
Syscall Data Generator
Generates synthetic and real syscall data for ML model training
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import subprocess
import os
import json

# Common Linux system calls with their typical characteristics
SYSCALL_CATALOG = {
    "read": {"category": "io", "base_latency": 0.5, "risk_level": 1},
    "write": {"category": "io", "base_latency": 0.8, "risk_level": 1},
    "open": {"category": "file", "base_latency": 1.2, "risk_level": 2},
    "close": {"category": "file", "base_latency": 0.3, "risk_level": 1},
    "stat": {"category": "file", "base_latency": 0.4, "risk_level": 1},
    "fstat": {"category": "file", "base_latency": 0.3, "risk_level": 1},
    "lstat": {"category": "file", "base_latency": 0.4, "risk_level": 1},
    "poll": {"category": "sync", "base_latency": 5.0, "risk_level": 2},
    "lseek": {"category": "file", "base_latency": 0.2, "risk_level": 1},
    "mmap": {"category": "memory", "base_latency": 2.5, "risk_level": 3},
    "mprotect": {"category": "memory", "base_latency": 1.5, "risk_level": 3},
    "munmap": {"category": "memory", "base_latency": 1.0, "risk_level": 2},
    "brk": {"category": "memory", "base_latency": 0.8, "risk_level": 2},
    "ioctl": {"category": "device", "base_latency": 3.0, "risk_level": 3},
    "access": {"category": "file", "base_latency": 0.5, "risk_level": 1},
    "pipe": {"category": "ipc", "base_latency": 1.5, "risk_level": 2},
    "select": {"category": "sync", "base_latency": 8.0, "risk_level": 2},
    "sched_yield": {"category": "process", "base_latency": 0.1, "risk_level": 1},
    "mremap": {"category": "memory", "base_latency": 3.0, "risk_level": 3},
    "msync": {"category": "memory", "base_latency": 10.0, "risk_level": 3},
    "mincore": {"category": "memory", "base_latency": 0.5, "risk_level": 1},
    "madvise": {"category": "memory", "base_latency": 0.4, "risk_level": 2},
    "socket": {"category": "network", "base_latency": 2.0, "risk_level": 3},
    "connect": {"category": "network", "base_latency": 50.0, "risk_level": 4},
    "accept": {"category": "network", "base_latency": 100.0, "risk_level": 4},
    "sendto": {"category": "network", "base_latency": 5.0, "risk_level": 3},
    "recvfrom": {"category": "network", "base_latency": 20.0, "risk_level": 3},
    "shutdown": {"category": "network", "base_latency": 1.0, "risk_level": 2},
    "bind": {"category": "network", "base_latency": 1.5, "risk_level": 3},
    "listen": {"category": "network", "base_latency": 0.5, "risk_level": 2},
    "clone": {"category": "process", "base_latency": 100.0, "risk_level": 5},
    "fork": {"category": "process", "base_latency": 150.0, "risk_level": 5},
    "vfork": {"category": "process", "base_latency": 80.0, "risk_level": 5},
    "execve": {"category": "process", "base_latency": 200.0, "risk_level": 5},
    "exit": {"category": "process", "base_latency": 50.0, "risk_level": 4},
    "wait4": {"category": "process", "base_latency": 500.0, "risk_level": 3},
    "kill": {"category": "process", "base_latency": 1.0, "risk_level": 4},
    "uname": {"category": "system", "base_latency": 0.2, "risk_level": 1},
    "getpid": {"category": "process", "base_latency": 0.1, "risk_level": 1},
    "getuid": {"category": "process", "base_latency": 0.1, "risk_level": 1},
    "getgid": {"category": "process", "base_latency": 0.1, "risk_level": 1},
    "gettimeofday": {"category": "time", "base_latency": 0.1, "risk_level": 1},
    "nanosleep": {"category": "time", "base_latency": 1000.0, "risk_level": 2},
    "futex": {"category": "sync", "base_latency": 5.0, "risk_level": 3},
    "epoll_wait": {"category": "sync", "base_latency": 10.0, "risk_level": 2},
    "epoll_create": {"category": "sync", "base_latency": 1.0, "risk_level": 2},
    "epoll_ctl": {"category": "sync", "base_latency": 0.5, "risk_level": 2},
}


class SyscallDataGenerator:
    """Generates synthetic syscall data for training ML models"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.syscalls = list(SYSCALL_CATALOG.keys())

    def generate_syscall_trace(self, num_calls=10000, process_type="mixed"):
        """
        Generate a synthetic syscall trace

        Args:
            num_calls: Number of syscalls to generate
            process_type: Type of process ('io_heavy', 'cpu_heavy', 'network', 'mixed')
        """
        # Define syscall distribution based on process type
        weights = self._get_weights_for_process_type(process_type)

        data = []
        timestamp = datetime.now()

        for i in range(num_calls):
            syscall = random.choices(self.syscalls, weights=weights)[0]
            info = SYSCALL_CATALOG[syscall]

            # Generate latency with some noise and occasional spikes
            base_latency = info["base_latency"]
            latency = self._generate_latency(base_latency)

            # Determine if this call is redundant (repeated within short window)
            is_redundant = self._check_redundancy(
                data[-10:] if len(data) >= 10 else data, syscall
            )

            # CPU and memory usage at this point
            cpu_usage = np.random.normal(30, 15)
            cpu_usage = np.clip(cpu_usage, 0, 100)
            memory_usage = np.random.normal(50, 20)
            memory_usage = np.clip(memory_usage, 0, 100)

            # Context switches
            context_switches = int(np.random.exponential(2))

            # Return value (0 = success, <0 = error)
            return_value = 0 if random.random() > 0.02 else -random.randint(1, 22)

            record = {
                "timestamp": timestamp.isoformat(),
                "syscall_id": i,
                "syscall_name": syscall,
                "category": info["category"],
                "latency_us": round(latency, 3),
                "base_latency_us": base_latency,
                "latency_ratio": round(latency / base_latency, 3),
                "cpu_usage": round(cpu_usage, 2),
                "memory_usage": round(memory_usage, 2),
                "context_switches": context_switches,
                "return_value": return_value,
                "is_redundant": is_redundant,
                "risk_level": info["risk_level"],
                "is_high_latency": latency > base_latency * 3,  # More than 3x expected
                "needs_optimization": (latency > base_latency * 2) or is_redundant,
            }

            data.append(record)
            timestamp += timedelta(microseconds=latency + random.randint(1, 100))

        return pd.DataFrame(data)

    def _get_weights_for_process_type(self, process_type):
        """Get syscall probability weights based on process type"""
        weights = []

        for syscall in self.syscalls:
            info = SYSCALL_CATALOG[syscall]
            category = info["category"]

            if process_type == "io_heavy":
                if category in ["io", "file"]:
                    weights.append(5.0)
                elif category == "memory":
                    weights.append(2.0)
                else:
                    weights.append(1.0)

            elif process_type == "cpu_heavy":
                if category in ["process", "memory"]:
                    weights.append(4.0)
                elif category == "sync":
                    weights.append(3.0)
                else:
                    weights.append(1.0)

            elif process_type == "network":
                if category == "network":
                    weights.append(6.0)
                elif category in ["sync", "io"]:
                    weights.append(2.0)
                else:
                    weights.append(1.0)

            else:  # mixed
                weights.append(1.0)

        # Normalize weights
        total = sum(weights)
        return [w / total for w in weights]

    def _generate_latency(self, base_latency):
        """Generate realistic latency with occasional spikes"""
        # Normal variation
        latency = np.random.lognormal(np.log(base_latency), 0.5)

        # Occasional spike (5% chance)
        if random.random() < 0.05:
            latency *= random.uniform(3, 10)

        # Very occasional major spike (1% chance)
        if random.random() < 0.01:
            latency *= random.uniform(10, 50)

        return max(0.01, latency)

    def _check_redundancy(self, recent_calls, current_syscall):
        """Check if current syscall is potentially redundant"""
        if not recent_calls:
            return False

        # Count same syscall in recent history
        same_count = sum(
            1 for r in recent_calls if r["syscall_name"] == current_syscall
        )

        # If more than 50% of recent calls are the same, it might be redundant
        return same_count > len(recent_calls) * 0.5

    def generate_training_dataset(self, samples_per_type=5000):
        """Generate a comprehensive training dataset"""
        dfs = []

        for process_type in ["io_heavy", "cpu_heavy", "network", "mixed"]:
            df = self.generate_syscall_trace(samples_per_type, process_type)
            df["process_type"] = process_type
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        return combined.sample(frac=1).reset_index(drop=True)  # Shuffle

    def generate_benchmark_data(self, num_processes=10, calls_per_process=1000):
        """Generate benchmark dataset with process-level metrics"""
        all_data = []

        for pid in range(num_processes):
            process_type = random.choice(["io_heavy", "cpu_heavy", "network", "mixed"])
            df = self.generate_syscall_trace(calls_per_process, process_type)
            df["pid"] = pid
            df["process_type"] = process_type
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)


def main():
    """Generate and save training data"""
    generator = SyscallDataGenerator()

    print("Generating training dataset...")
    training_data = generator.generate_training_dataset(samples_per_type=10000)
    training_data.to_csv("training_data.csv", index=False)
    print(f"Saved training_data.csv with {len(training_data)} records")

    print("\nGenerating benchmark dataset...")
    benchmark_data = generator.generate_benchmark_data(
        num_processes=20, calls_per_process=500
    )
    benchmark_data.to_csv("benchmark_data.csv", index=False)
    print(f"Saved benchmark_data.csv with {len(benchmark_data)} records")

    # Print summary statistics
    print("\n=== Training Data Summary ===")
    print(f"Total syscalls: {len(training_data)}")
    print(f"Unique syscalls: {training_data['syscall_name'].nunique()}")
    print(
        f"High latency calls: {training_data['is_high_latency'].sum()} ({training_data['is_high_latency'].mean()*100:.2f}%)"
    )
    print(
        f"Redundant calls: {training_data['is_redundant'].sum()} ({training_data['is_redundant'].mean()*100:.2f}%)"
    )
    print(
        f"Needs optimization: {training_data['needs_optimization'].sum()} ({training_data['needs_optimization'].mean()*100:.2f}%)"
    )


if __name__ == "__main__":
    main()
