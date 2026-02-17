"""
Strace Output Parser
Parses real strace output files and converts them to ML-compatible DataFrames

Supports multiple strace formats:
  - strace -T         (shows time spent in each syscall)
  - strace -tt -T     (with timestamps and durations)
  - strace -c         (summary/statistics mode)
  - strace -e trace=  (filtered traces)

Usage:
  parser = StraceParser()
  df = parser.parse_file("strace_output.txt")
  df = parser.parse_text(strace_text)
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.syscall_data_generator import SYSCALL_CATALOG


class StraceParser:
    """
    Parser for strace output files.
    Converts raw strace logs into structured DataFrames for ML analysis.
    """

    # Regex patterns for parsing different strace output formats

    # Pattern: PID  syscall(args) = return_value <time_spent>
    # Example: 12345 read(3, "data", 1024) = 512 <0.000123>
    PATTERN_WITH_PID = re.compile(
        r"^(\d+)\s+"  # PID
        r"(\w+)\("  # syscall name
        r"(.*?)\)\s*"  # arguments
        r"=\s*([-\d?]+|0x[0-9a-f]+)"  # return value
        r"(?:\s+(\S+))?"  # optional errno description
        r"(?:\s+<([\d.]+)>)?"  # optional time spent <seconds>
        r"\s*$"
    )

    # Pattern: HH:MM:SS.microsecs syscall(args) = return_value <time_spent>
    # Example: 14:12:05.123456 read(3, "data", 1024) = 512 <0.000123>
    PATTERN_WITH_TIME = re.compile(
        r"^(\d{2}:\d{2}:\d{2}\.\d+)\s+"  # timestamp
        r"(\w+)\("  # syscall name
        r"(.*?)\)\s*"  # arguments
        r"=\s*([-\d?]+|0x[0-9a-f]+)"  # return value
        r"(?:\s+(\S+))?"  # optional errno description
        r"(?:\s+<([\d.]+)>)?"  # optional time spent
        r"\s*$"
    )

    # Pattern: PID HH:MM:SS.microsecs syscall(args) = return_value <time_spent>
    PATTERN_PID_TIME = re.compile(
        r"^(\d+)\s+"  # PID
        r"(\d{2}:\d{2}:\d{2}\.\d+)\s+"  # timestamp
        r"(\w+)\("  # syscall name
        r"(.*?)\)\s*"  # arguments
        r"=\s*([-\d?]+|0x[0-9a-f]+)"  # return value
        r"(?:\s+(\S+))?"  # optional errno description
        r"(?:\s+<([\d.]+)>)?"  # optional time spent
        r"\s*$"
    )

    # Pattern: simple — syscall(args) = return_value
    # Example: read(3, "data", 1024) = 512
    PATTERN_SIMPLE = re.compile(
        r"^(\w+)\("  # syscall name
        r"(.*?)\)\s*"  # arguments
        r"=\s*([-\d?]+|0x[0-9a-f]+)"  # return value
        r"(?:\s+(\S+))?"  # optional errno description
        r"(?:\s+<([\d.]+)>)?"  # optional time spent
        r"\s*$"
    )

    # Pattern for strace -c (summary) output
    # % time     seconds  usecs/call     calls    errors syscall
    PATTERN_SUMMARY = re.compile(
        r"^\s*([\d.]+)\s+"  # % time
        r"([\d.]+)\s+"  # seconds
        r"(\d+)\s+"  # usecs/call
        r"(\d+)\s+"  # calls
        r"(\d*)\s+"  # errors (optional)
        r"(\w+)\s*$"  # syscall name
    )

    # Signals pattern (to skip)
    PATTERN_SIGNAL = re.compile(r"^---\s+(\w+)\s+")

    # Category mapping for unknown syscalls
    CATEGORY_MAP = {
        "io": [
            "read",
            "write",
            "pread64",
            "pwrite64",
            "readv",
            "writev",
            "sendfile",
            "splice",
            "tee",
            "copy_file_range",
        ],
        "file": [
            "open",
            "openat",
            "close",
            "stat",
            "fstat",
            "lstat",
            "newfstatat",
            "access",
            "faccessat",
            "lseek",
            "dup",
            "dup2",
            "dup3",
            "fcntl",
            "flock",
            "chmod",
            "fchmod",
            "chown",
            "fchown",
            "rename",
            "renameat",
            "renameat2",
            "unlink",
            "unlinkat",
            "mkdir",
            "mkdirat",
            "rmdir",
            "getcwd",
            "chdir",
            "fchdir",
            "readlink",
            "readlinkat",
            "statfs",
            "fstatfs",
            "truncate",
            "ftruncate",
            "getdents",
            "getdents64",
            "utimensat",
        ],
        "memory": [
            "mmap",
            "mmap2",
            "munmap",
            "mprotect",
            "brk",
            "mremap",
            "msync",
            "mincore",
            "madvise",
            "mlock",
            "munlock",
            "shmget",
            "shmat",
            "shmctl",
            "shmdt",
        ],
        "network": [
            "socket",
            "connect",
            "accept",
            "accept4",
            "bind",
            "listen",
            "sendto",
            "recvfrom",
            "sendmsg",
            "recvmsg",
            "shutdown",
            "setsockopt",
            "getsockopt",
            "getsockname",
            "getpeername",
            "socketpair",
        ],
        "process": [
            "clone",
            "clone3",
            "fork",
            "vfork",
            "execve",
            "execveat",
            "exit",
            "exit_group",
            "wait4",
            "waitid",
            "kill",
            "tkill",
            "tgkill",
            "getpid",
            "getppid",
            "gettid",
            "getuid",
            "geteuid",
            "getgid",
            "getegid",
            "setuid",
            "setgid",
            "getgroups",
            "setgroups",
            "prctl",
            "arch_prctl",
            "set_tid_address",
            "set_robust_list",
            "get_robust_list",
            "sched_yield",
            "sched_getaffinity",
            "sched_setaffinity",
        ],
        "sync": [
            "poll",
            "ppoll",
            "select",
            "pselect6",
            "epoll_create",
            "epoll_create1",
            "epoll_ctl",
            "epoll_wait",
            "epoll_pwait",
            "futex",
            "eventfd",
            "eventfd2",
            "signalfd",
            "signalfd4",
            "timerfd_create",
            "timerfd_settime",
            "timerfd_gettime",
        ],
        "signal": [
            "rt_sigaction",
            "rt_sigprocmask",
            "rt_sigreturn",
            "rt_sigpending",
            "rt_sigtimedwait",
            "rt_sigsuspend",
            "sigaltstack",
            "signal",
        ],
        "system": ["uname", "sysinfo", "syslog", "ioctl", "getrandom"],
        "time": [
            "gettimeofday",
            "clock_gettime",
            "clock_getres",
            "clock_nanosleep",
            "nanosleep",
            "time",
            "times",
        ],
        "ipc": [
            "pipe",
            "pipe2",
            "msgget",
            "msgsnd",
            "msgrcv",
            "msgctl",
            "semget",
            "semop",
            "semctl",
        ],
    }

    def __init__(self):
        self.parsed_count = 0
        self.skipped_count = 0
        self.errors = []

    def _get_category(self, syscall_name):
        """Determine category for a syscall"""
        # Check SYSCALL_CATALOG first
        if syscall_name in SYSCALL_CATALOG:
            return SYSCALL_CATALOG[syscall_name]["category"]

        # Check our extended category map
        for category, syscalls in self.CATEGORY_MAP.items():
            if syscall_name in syscalls:
                return category

        return "other"

    def _get_base_latency(self, syscall_name):
        """Get base latency for a syscall in microseconds"""
        if syscall_name in SYSCALL_CATALOG:
            return SYSCALL_CATALOG[syscall_name]["base_latency"]

        # Estimate based on category
        category_defaults = {
            "io": 1.0,
            "file": 0.8,
            "memory": 2.0,
            "network": 10.0,
            "process": 50.0,
            "sync": 5.0,
            "signal": 0.5,
            "system": 0.5,
            "time": 0.2,
            "ipc": 2.0,
            "other": 1.0,
        }
        category = self._get_category(syscall_name)
        return category_defaults.get(category, 1.0)

    def _get_risk_level(self, syscall_name):
        """Get risk level for a syscall"""
        if syscall_name in SYSCALL_CATALOG:
            return SYSCALL_CATALOG[syscall_name]["risk_level"]

        # Estimate based on category
        category_risks = {
            "io": 1,
            "file": 2,
            "memory": 3,
            "network": 3,
            "process": 4,
            "sync": 2,
            "signal": 2,
            "system": 2,
            "time": 1,
            "ipc": 2,
            "other": 2,
        }
        category = self._get_category(syscall_name)
        return category_risks.get(category, 2)

    def parse_line(self, line):
        """
        Parse a single line of strace output.
        Returns a dict with parsed fields or None if unparseable.
        """
        line = line.strip()

        # Skip empty lines, signals, and unfinished/resumed calls
        if not line or line.startswith("---") or line.startswith("+++"):
            return None
        if "<unfinished" in line or "resumed>" in line:
            return None

        record = {
            "pid": None,
            "timestamp": None,
            "syscall_name": None,
            "args": None,
            "return_value": None,
            "errno": None,
            "duration_seconds": None,
        }

        # Try PID + timestamp pattern first (most specific)
        m = self.PATTERN_PID_TIME.match(line)
        if m:
            record["pid"] = int(m.group(1))
            record["timestamp"] = m.group(2)
            record["syscall_name"] = m.group(3)
            record["args"] = m.group(4)
            record["return_value"] = m.group(5)
            record["errno"] = m.group(6)
            record["duration_seconds"] = float(m.group(7)) if m.group(7) else None
            return record

        # Try PID pattern
        m = self.PATTERN_WITH_PID.match(line)
        if m:
            record["pid"] = int(m.group(1))
            record["syscall_name"] = m.group(2)
            record["args"] = m.group(3)
            record["return_value"] = m.group(4)
            record["errno"] = m.group(5)
            record["duration_seconds"] = float(m.group(6)) if m.group(6) else None
            return record

        # Try timestamp pattern
        m = self.PATTERN_WITH_TIME.match(line)
        if m:
            record["timestamp"] = m.group(1)
            record["syscall_name"] = m.group(2)
            record["args"] = m.group(3)
            record["return_value"] = m.group(4)
            record["errno"] = m.group(5)
            record["duration_seconds"] = float(m.group(6)) if m.group(6) else None
            return record

        # Try simple pattern
        m = self.PATTERN_SIMPLE.match(line)
        if m:
            record["syscall_name"] = m.group(1)
            record["args"] = m.group(2)
            record["return_value"] = m.group(3)
            record["errno"] = m.group(4)
            record["duration_seconds"] = float(m.group(5)) if m.group(5) else None
            return record

        return None

    def parse_summary_line(self, line):
        """Parse a line from strace -c (summary) output"""
        m = self.PATTERN_SUMMARY.match(line.strip())
        if m:
            return {
                "pct_time": float(m.group(1)),
                "seconds": float(m.group(2)),
                "usecs_per_call": int(m.group(3)),
                "calls": int(m.group(4)),
                "errors": int(m.group(5)) if m.group(5) else 0,
                "syscall_name": m.group(6),
            }
        return None

    def parse_text(self, text):
        """
        Parse raw strace text output into a DataFrame.

        Args:
            text: Raw strace output string

        Returns:
            pd.DataFrame with columns compatible with ML models
        """
        lines = text.strip().split("\n")
        records = []
        summary_records = []
        self.parsed_count = 0
        self.skipped_count = 0
        self.errors = []

        # Detect if this is summary (-c) output
        is_summary = False
        for line in lines:
            if "% time" in line and "usecs/call" in line:
                is_summary = True
                break

        if is_summary:
            return self._parse_summary(lines)

        # Parse trace output
        recent_syscalls = []

        for line_num, line in enumerate(lines, 1):
            try:
                parsed = self.parse_line(line)
                if parsed and parsed["syscall_name"]:
                    self.parsed_count += 1

                    # Calculate latency in microseconds
                    if parsed["duration_seconds"] is not None:
                        latency_us = parsed["duration_seconds"] * 1_000_000
                    else:
                        # Estimate based on syscall type
                        base = self._get_base_latency(parsed["syscall_name"])
                        latency_us = base * np.random.lognormal(0, 0.3)

                    base_latency = self._get_base_latency(parsed["syscall_name"])
                    category = self._get_category(parsed["syscall_name"])
                    risk_level = self._get_risk_level(parsed["syscall_name"])

                    # Detect return error
                    ret_val = parsed["return_value"]
                    try:
                        return_code = int(ret_val)
                    except (ValueError, TypeError):
                        return_code = 0

                    # Check for redundancy
                    is_redundant = self._check_redundancy(
                        recent_syscalls, parsed["syscall_name"]
                    )
                    recent_syscalls.append(parsed["syscall_name"])
                    if len(recent_syscalls) > 10:
                        recent_syscalls.pop(0)

                    # Calculate derived metrics
                    latency_ratio = (
                        latency_us / base_latency if base_latency > 0 else 1.0
                    )
                    is_high_latency = latency_us > base_latency * 3
                    needs_optimization = is_high_latency or is_redundant

                    # Simulated resource usage (strace doesn't capture these directly)
                    # In production, you'd combine with /proc/<pid>/stat
                    cpu_usage = np.clip(np.random.normal(30, 15), 0, 100)
                    memory_usage = np.clip(np.random.normal(50, 20), 0, 100)
                    context_switches = int(np.random.exponential(2))

                    record = {
                        "timestamp": parsed["timestamp"] or datetime.now().isoformat(),
                        "syscall_id": self.parsed_count - 1,
                        "pid": parsed["pid"] or 0,
                        "syscall_name": parsed["syscall_name"],
                        "category": category,
                        "args_summary": (parsed["args"] or "")[:100],
                        "latency_us": round(latency_us, 3),
                        "base_latency_us": base_latency,
                        "latency_ratio": round(latency_ratio, 3),
                        "cpu_usage": round(cpu_usage, 2),
                        "memory_usage": round(memory_usage, 2),
                        "context_switches": context_switches,
                        "return_value": return_code,
                        "errno": parsed["errno"] or "",
                        "is_redundant": is_redundant,
                        "risk_level": risk_level,
                        "is_high_latency": is_high_latency,
                        "needs_optimization": needs_optimization,
                        "source": "strace",
                    }

                    records.append(record)
                else:
                    self.skipped_count += 1

            except Exception as e:
                self.skipped_count += 1
                self.errors.append(f"Line {line_num}: {str(e)}")

        if not records:
            # Return an empty DataFrame with the right columns
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "syscall_id",
                    "pid",
                    "syscall_name",
                    "category",
                    "args_summary",
                    "latency_us",
                    "base_latency_us",
                    "latency_ratio",
                    "cpu_usage",
                    "memory_usage",
                    "context_switches",
                    "return_value",
                    "errno",
                    "is_redundant",
                    "risk_level",
                    "is_high_latency",
                    "needs_optimization",
                    "source",
                ]
            )

        return pd.DataFrame(records)

    def _parse_summary(self, lines):
        """Parse strace -c summary output into a DataFrame"""
        records = []
        parsing = False

        for line in lines:
            if "------" in line:
                parsing = not parsing
                continue

            if parsing:
                parsed = self.parse_summary_line(line)
                if parsed:
                    syscall_name = parsed["syscall_name"]
                    base_latency = self._get_base_latency(syscall_name)
                    category = self._get_category(syscall_name)
                    risk_level = self._get_risk_level(syscall_name)
                    latency_us = parsed["usecs_per_call"]

                    # Expand summary into individual-call-like records
                    for i in range(parsed["calls"]):
                        actual_latency = latency_us * np.random.lognormal(0, 0.2)
                        is_error = i < parsed["errors"]

                        record = {
                            "timestamp": datetime.now().isoformat(),
                            "syscall_id": len(records),
                            "pid": 0,
                            "syscall_name": syscall_name,
                            "category": category,
                            "args_summary": "",
                            "latency_us": round(actual_latency, 3),
                            "base_latency_us": base_latency,
                            "latency_ratio": round(
                                actual_latency / max(base_latency, 0.01), 3
                            ),
                            "cpu_usage": round(
                                np.clip(np.random.normal(30, 15), 0, 100), 2
                            ),
                            "memory_usage": round(
                                np.clip(np.random.normal(50, 20), 0, 100), 2
                            ),
                            "context_switches": int(np.random.exponential(2)),
                            "return_value": -1 if is_error else 0,
                            "errno": "ERROR" if is_error else "",
                            "is_redundant": False,
                            "risk_level": risk_level,
                            "is_high_latency": actual_latency > base_latency * 3,
                            "needs_optimization": actual_latency > base_latency * 2,
                            "source": "strace_summary",
                        }
                        records.append(record)

                    self.parsed_count += 1

        return pd.DataFrame(records) if records else pd.DataFrame()

    def _check_redundancy(self, recent, current):
        """Check if a syscall is redundant based on recent history"""
        if len(recent) < 3:
            return False
        same_count = sum(1 for s in recent[-5:] if s == current)
        return same_count >= 3

    def parse_file(self, filepath):
        """
        Parse a strace output file.

        Args:
            filepath: Path to strace output file

        Returns:
            pd.DataFrame
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Strace file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        return self.parse_text(text)

    def get_parse_stats(self):
        """Get statistics about the last parse operation"""
        return {
            "parsed_lines": self.parsed_count,
            "skipped_lines": self.skipped_count,
            "errors": self.errors[:10],  # First 10 errors
            "total_errors": len(self.errors),
        }


def main():
    """Test the parser with sample data"""
    parser = StraceParser()

    # Test with sample strace output
    sample = """14:12:05.123456 read(3, "hello", 1024) = 5 <0.000012>
14:12:05.123500 write(1, "hello", 5) = 5 <0.000008>
14:12:05.123600 openat(AT_FDCWD, "/etc/passwd", O_RDONLY) = 4 <0.000045>
14:12:05.123700 fstat(4, {st_mode=S_IFREG|0644, st_size=2773, ...}) = 0 <0.000006>
14:12:05.123800 read(4, "root:x:0:0:root:/root:/bin/bash\\n"..., 4096) = 2773 <0.000015>
14:12:05.123900 close(4) = 0 <0.000005>
14:12:05.124000 mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f1234560000 <0.000020>
14:12:05.124100 socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 5 <0.000030>
14:12:05.124200 connect(5, {sa_family=AF_INET, sin_port=htons(80), sin_addr=inet_addr("192.168.1.1")}, 16) = -1 EINPROGRESS (Operation now in progress) <0.000350>
14:12:05.125000 epoll_wait(3, [{EPOLLOUT, {u32=5}}], 128, 1000) = 1 <0.000800>
14:12:05.126000 write(5, "GET / HTTP/1.1\\r\\nHost: example.com\\r\\n\\r\\n", 38) = 38 <0.000025>
14:12:05.126100 read(5, "HTTP/1.1 200 OK\\r\\n"..., 8192) = 1024 <0.005000>
14:12:05.131200 close(5) = 0 <0.000010>
14:12:05.131300 munmap(0x7f1234560000, 4096) = 0 <0.000015>
14:12:05.131400 futex(0x7f1234560100, FUTEX_WAKE_PRIVATE, 1) = 0 <0.000008>"""

    print("Parsing sample strace output...")
    df = parser.parse_text(sample)

    print(f"\nParsed {len(df)} syscalls")
    print(f"Stats: {parser.get_parse_stats()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:")
    print(
        df[
            [
                "syscall_name",
                "category",
                "latency_us",
                "is_high_latency",
                "is_redundant",
            ]
        ].to_string()
    )

    print(f"\n=== Summary ===")
    print(f"Unique syscalls: {df['syscall_name'].nunique()}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"High latency: {df['is_high_latency'].sum()}")
    print(f"Avg latency: {df['latency_us'].mean():.2f} μs")


if __name__ == "__main__":
    main()
