"""
Strace Runner
Runs strace on commands/processes and captures output.
Supports Linux (strace) and macOS (dtruss/dtrace).

Usage:
  runner = StraceRunner()
  result = runner.trace_command("ls -la")
  result = runner.trace_pid(12345, duration=10)
"""

import subprocess
import tempfile
import os
import sys
import platform
import signal
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strace.strace_parser import StraceParser


class StraceRunner:
    """
    Runs strace/dtruss on commands or processes and returns parsed results.
    """

    def __init__(self):
        self.parser = StraceParser()
        self.system = platform.system().lower()
        self._detect_tool()

    def _detect_tool(self):
        """Detect which tracing tool is available"""
        self.tool = None
        self.tool_name = None

        if self.system == "linux":
            # Check for strace
            try:
                subprocess.run(["strace", "--version"], capture_output=True, timeout=5)
                self.tool = "strace"
                self.tool_name = "strace"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Check for ltrace as fallback
            if not self.tool:
                try:
                    subprocess.run(
                        ["ltrace", "--version"], capture_output=True, timeout=5
                    )
                    self.tool = "ltrace"
                    self.tool_name = "ltrace"
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

        elif self.system == "darwin":
            # macOS: check for dtruss (requires SIP disabled or root)
            try:
                subprocess.run(["which", "dtruss"], capture_output=True, timeout=5)
                self.tool = "dtruss"
                self.tool_name = "dtruss (macOS)"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # dtrace as alternative
            if not self.tool:
                try:
                    subprocess.run(["which", "dtrace"], capture_output=True, timeout=5)
                    self.tool = "dtrace"
                    self.tool_name = "dtrace (macOS)"
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

    def is_available(self):
        """Check if any tracing tool is available"""
        return self.tool is not None

    def get_tool_info(self):
        """Get information about the available tracing tool"""
        return {
            "system": self.system,
            "tool": self.tool,
            "tool_name": self.tool_name,
            "available": self.tool is not None,
            "requires_root": self.system == "darwin",  # dtruss requires sudo on macOS
            "note": self._get_tool_note(),
        }

    def _get_tool_note(self):
        """Get a note about the tracing tool"""
        if self.system == "darwin":
            return (
                "macOS requires 'dtruss' or 'dtrace' which need root privileges. "
                "System Integrity Protection (SIP) may need to be disabled. "
                "Alternatively, upload an strace output file from a Linux system."
            )
        elif self.system == "linux":
            if self.tool:
                return f"{self.tool} is available and ready to use."
            else:
                return "Install strace: sudo apt-get install strace (Debian/Ubuntu) or sudo yum install strace (CentOS/RHEL)"
        else:
            return f"Unsupported system: {self.system}. Upload strace output files instead."

    def trace_command(self, command, timeout=30):
        """
        Run strace on a command and return parsed results.

        Args:
            command: Command string to trace (e.g., "ls -la /tmp")
            timeout: Maximum time to run in seconds

        Returns:
            dict with 'success', 'data' (DataFrame), 'raw_output', 'stats'
        """
        if not self.tool:
            return {
                "success": False,
                "error": f"No tracing tool available on {self.system}. {self._get_tool_note()}",
                "data": None,
                "raw_output": "",
                "stats": {},
            }

        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".strace", delete=False, prefix="syscall_"
            ) as tmp:
                output_file = tmp.name

            # Build command based on tool
            if self.tool == "strace":
                trace_cmd = [
                    "strace",
                    "-T",  # Show time spent in syscalls
                    "-tt",  # Show timestamps with microseconds
                    "-f",  # Follow forks
                    "-o",
                    output_file,  # Output to file
                ] + command.split()

            elif self.tool == "dtruss":
                # dtruss requires sudo on macOS
                trace_cmd = [
                    "sudo",
                    "dtruss",
                    "-f",  # Follow children
                ] + command.split()
                # dtruss outputs to stderr, we'll capture it

            elif self.tool == "dtrace":
                # Use a simple dtrace script for syscall tracing
                dtrace_script = (
                    "syscall:::entry { self->ts = timestamp; } "
                    "syscall:::return /self->ts/ "
                    '{ printf("%s(%d) = %d <%d>\\n", probefunc, pid, arg1, '
                    "(timestamp - self->ts) / 1000); self->ts = 0; }"
                )
                trace_cmd = ["sudo", "dtrace", "-n", dtrace_script, "-c", command]

            # Run the command
            start_time = time.time()
            result = subprocess.run(
                trace_cmd, capture_output=True, text=True, timeout=timeout
            )
            elapsed = time.time() - start_time

            # Read strace output
            if self.tool == "strace" and os.path.exists(output_file):
                with open(output_file, "r") as f:
                    raw_output = f.read()
            else:
                # dtruss/dtrace output goes to stderr
                raw_output = result.stderr

            # Parse the output
            df = self.parser.parse_text(raw_output)
            stats = self.parser.get_parse_stats()
            stats["elapsed_seconds"] = round(elapsed, 3)
            stats["command"] = command
            stats["tool_used"] = self.tool

            # Clean up temp file
            if os.path.exists(output_file):
                os.unlink(output_file)

            return {
                "success": True,
                "data": df,
                "raw_output": raw_output,
                "stats": stats,
                "command_stdout": result.stdout,
                "command_returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "data": None,
                "raw_output": "",
                "stats": {},
            }
        except PermissionError:
            return {
                "success": False,
                "error": "Permission denied. Tracing may require root/sudo privileges.",
                "data": None,
                "raw_output": "",
                "stats": {},
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "raw_output": "",
                "stats": {},
            }

    def trace_pid(self, pid, duration=10):
        """
        Attach strace to a running process by PID.

        Args:
            pid: Process ID to trace
            duration: How long to trace in seconds

        Returns:
            dict with parsed results
        """
        if not self.tool:
            return {
                "success": False,
                "error": f"No tracing tool available. {self._get_tool_note()}",
                "data": None,
                "raw_output": "",
                "stats": {},
            }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".strace", delete=False, prefix="syscall_pid_"
            ) as tmp:
                output_file = tmp.name

            if self.tool == "strace":
                trace_cmd = [
                    "strace",
                    "-T",
                    "-tt",
                    "-f",
                    "-p",
                    str(pid),
                    "-o",
                    output_file,
                ]
            elif self.tool == "dtruss":
                trace_cmd = ["sudo", "dtruss", "-f", "-p", str(pid)]
            else:
                return {
                    "success": False,
                    "error": f"{self.tool} does not support PID tracing easily",
                    "data": None,
                    "raw_output": "",
                    "stats": {},
                }

            # Start the trace
            proc = subprocess.Popen(
                trace_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for the specified duration, then stop
            time.sleep(duration)
            proc.send_signal(signal.SIGINT)
            stdout, stderr = proc.communicate(timeout=5)

            # Read output
            if self.tool == "strace" and os.path.exists(output_file):
                with open(output_file, "r") as f:
                    raw_output = f.read()
            else:
                raw_output = stderr

            df = self.parser.parse_text(raw_output)
            stats = self.parser.get_parse_stats()
            stats["pid"] = pid
            stats["duration"] = duration
            stats["tool_used"] = self.tool

            if os.path.exists(output_file):
                os.unlink(output_file)

            return {
                "success": True,
                "data": df,
                "raw_output": raw_output,
                "stats": stats,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "raw_output": "",
                "stats": {},
            }


def main():
    """Test the strace runner"""
    runner = StraceRunner()

    print("=== Strace Runner Info ===")
    info = runner.get_tool_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    if runner.is_available():
        print("\n=== Tracing 'ls -la /tmp' ===")
        result = runner.trace_command("ls -la /tmp", timeout=10)
        if result["success"]:
            print(f"Parsed {len(result['data'])} syscalls")
            print(f"Stats: {result['stats']}")
        else:
            print(f"Error: {result['error']}")
    else:
        print(
            "\nNo tracing tool available. Use StraceParser to parse existing strace files."
        )


if __name__ == "__main__":
    main()
