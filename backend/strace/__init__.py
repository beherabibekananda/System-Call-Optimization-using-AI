"""
Strace Integration Module
- Parse real strace output files
- Run strace/dtruss on processes
- Convert strace data into ML-compatible DataFrames
"""

from .strace_parser import StraceParser
from .strace_runner import StraceRunner
