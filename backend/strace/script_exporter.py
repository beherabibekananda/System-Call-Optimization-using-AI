"""
Script Exporter
Exports optimization recommendations as executable shell scripts
and reports in various formats (JSON, CSV, Markdown).
"""

import json
import os
from datetime import datetime


class ScriptExporter:
    """
    Exports ML optimization recommendations as shell scripts,
    JSON reports, and markdown reports.
    """

    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "exports")
        os.makedirs(self.output_dir, exist_ok=True)

    def export_optimization_script(self, recommendations, analysis=None, filename=None):
        """
        Generate a shell script with optimization commands based on recommendations.

        Args:
            recommendations: List of recommendation dicts from the ML models
            analysis: Optional analysis dict for context
            filename: Optional custom filename

        Returns:
            dict with 'success', 'filepath', 'content'
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimize_{timestamp}.sh"

        filepath = os.path.join(self.output_dir, filename)

        lines = [
            "#!/bin/bash",
            "#",
            "# SysCall AI - System Call Optimization Script",
            f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            "# WARNING: Review all commands before executing!",
            "#",
            "# Usage: chmod +x optimize.sh && sudo ./optimize.sh",
            "#",
            "",
            "set -euo pipefail",
            "",
            'echo "=========================================="',
            'echo "  SysCall AI - Optimization Script"',
            'echo "=========================================="',
            'echo ""',
            "",
        ]

        # Add analysis summary if available
        if analysis:
            lines.extend(
                [
                    "# === Analysis Summary ===",
                    f'# Total syscalls analyzed: {analysis.get("total_syscalls", "N/A")}',
                    f'# High latency calls: {analysis.get("high_latency_count", "N/A")}',
                    f'# Optimization candidates: {analysis.get("optimization_candidates", "N/A")}',
                    f'# Anomalies detected: {analysis.get("anomalies_detected", "N/A")}',
                    f'# Average latency: {analysis.get("avg_latency", "N/A")} μs',
                    "",
                ]
            )

        # Group recommendations by type
        high_latency_recs = [
            r for r in recommendations if r.get("type") == "high_latency"
        ]
        redundancy_recs = [r for r in recommendations if r.get("type") == "redundancy"]
        anomaly_recs = [r for r in recommendations if r.get("type") == "anomaly"]
        resource_recs = [r for r in recommendations if "actions" in r]

        # High latency optimizations
        if high_latency_recs:
            lines.extend(
                [
                    "# =============================================",
                    "# 1. HIGH LATENCY SYSCALL OPTIMIZATIONS",
                    "# =============================================",
                    "",
                    'echo "[1/4] Applying high-latency optimizations..."',
                    "",
                ]
            )

            for rec in high_latency_recs:
                syscall = rec.get("syscall", "unknown")
                avg_lat = rec.get("avg_latency", 0)
                count = rec.get("count", 0)

                lines.extend(
                    [
                        f"# Syscall: {syscall}",
                        f"# Issue: {count} calls with avg latency {avg_lat:.2f} μs",
                        f'# Recommendation: {rec.get("recommendation", "")}',
                    ]
                )

                # Generate specific commands based on syscall type
                if syscall in ("read", "write", "pread64", "pwrite64"):
                    lines.extend(
                        [
                            f"# Tune I/O scheduler for better {syscall} performance",
                            "if [ -f /sys/block/sda/queue/scheduler ]; then",
                            '    echo "deadline" | sudo tee /sys/block/sda/queue/scheduler',
                            '    echo "  ✓ I/O scheduler set to deadline"',
                            "fi",
                            "",
                            "# Increase read-ahead buffer",
                            "if [ -f /sys/block/sda/queue/read_ahead_kb ]; then",
                            "    echo 256 | sudo tee /sys/block/sda/queue/read_ahead_kb",
                            '    echo "  ✓ Read-ahead buffer increased to 256KB"',
                            "fi",
                            "",
                        ]
                    )
                elif syscall in ("mmap", "munmap", "mprotect", "brk"):
                    lines.extend(
                        [
                            "# Tune memory management for better performance",
                            "sudo sysctl -w vm.swappiness=10",
                            'echo "  ✓ Swappiness reduced to 10"',
                            "",
                            "# Enable transparent huge pages for large allocations",
                            "if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then",
                            '    echo "madvise" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled',
                            '    echo "  ✓ Transparent huge pages set to madvise"',
                            "fi",
                            "",
                        ]
                    )
                elif syscall in ("connect", "accept", "socket", "bind"):
                    lines.extend(
                        [
                            "# Tune network stack for better performance",
                            "sudo sysctl -w net.core.somaxconn=65535",
                            "sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535",
                            "sudo sysctl -w net.core.netdev_max_backlog=65535",
                            'echo "  ✓ Network backlog buffers increased"',
                            "",
                            "# Enable TCP Fast Open",
                            "sudo sysctl -w net.ipv4.tcp_fastopen=3",
                            'echo "  ✓ TCP Fast Open enabled"',
                            "",
                        ]
                    )
                elif syscall in ("epoll_wait", "poll", "select", "futex"):
                    lines.extend(
                        [
                            "# Tune synchronization primitives",
                            "sudo sysctl -w kernel.sched_min_granularity_ns=1000000",
                            "sudo sysctl -w kernel.sched_wakeup_granularity_ns=1500000",
                            'echo "  ✓ Scheduler granularity tuned"',
                            "",
                        ]
                    )
                elif syscall in ("fork", "clone", "execve", "vfork"):
                    lines.extend(
                        [
                            "# Tune process creation performance",
                            "sudo sysctl -w kernel.pid_max=4194304",
                            "sudo sysctl -w kernel.threads-max=256000",
                            'echo "  ✓ Process limits increased"',
                            "",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            f'echo "  ⚠ Manual review needed for {syscall} optimization"',
                            "",
                        ]
                    )

        # Redundancy reduction
        if redundancy_recs:
            lines.extend(
                [
                    "# =============================================",
                    "# 2. REDUNDANCY REDUCTION",
                    "# =============================================",
                    "",
                    'echo "[2/4] Addressing redundant syscall patterns..."',
                    "",
                ]
            )

            for rec in redundancy_recs:
                syscall = rec.get("syscall", "unknown")
                count = rec.get("count", 0)
                lines.extend(
                    [
                        f"# {syscall}: {count} redundant calls detected",
                        f'# {rec.get("recommendation", "")}',
                        f'echo "  ℹ {syscall}: {count} redundant calls to optimize in application code"',
                        "",
                    ]
                )

        # Resource optimizations
        if resource_recs:
            lines.extend(
                [
                    "# =============================================",
                    "# 3. RESOURCE ALLOCATION RECOMMENDATIONS",
                    "# =============================================",
                    "",
                    'echo "[3/4] Applying resource allocation changes..."',
                    "",
                ]
            )

            for rec in resource_recs:
                pid = rec.get("pid", "N/A")
                actions = rec.get("actions", [])

                lines.append(
                    f'# PID {pid} (Priority: {rec.get("priority_score", 0):.2f})'
                )

                for action in actions:
                    action_type = action.get("type", "")
                    reason = action.get("reason", "")

                    if action_type == "increase_cpu":
                        lines.extend(
                            [
                                f"# Reason: {reason}",
                                f"# Consider: taskset or cpuset to pin PID {pid} to more cores",
                                f'echo "  ℹ PID {pid}: Consider CPU affinity adjustment"',
                            ]
                        )
                    elif action_type == "decrease_cpu":
                        lines.extend(
                            [
                                f"# Reason: {reason}",
                                f"# Consider: cgroups to limit CPU for PID {pid}",
                                f'echo "  ℹ PID {pid}: Consider CPU limit via cgroups"',
                            ]
                        )
                    elif action_type == "priority_boost":
                        lines.extend(
                            [
                                f"# Reason: {reason}",
                                f"# sudo renice -5 -p {pid}",
                                f'echo "  ℹ PID {pid}: Consider priority boost (renice)"',
                            ]
                        )
                    elif action_type == "code_optimization":
                        lines.extend(
                            [
                                f"# Reason: {reason}",
                                f'echo "  ℹ PID {pid}: Code-level optimization recommended"',
                            ]
                        )

                lines.append("")

        # Anomaly investigation
        if anomaly_recs:
            lines.extend(
                [
                    "# =============================================",
                    "# 4. ANOMALY INVESTIGATION",
                    "# =============================================",
                    "",
                    'echo "[4/4] Investigating anomalous patterns..."',
                    "",
                ]
            )

            for rec in anomaly_recs:
                lines.extend(
                    [
                        f'# {rec.get("recommendation", "")}',
                        f'echo "  ⚠ {rec.get("count", 0)} anomalous patterns detected - manual investigation recommended"',
                        "",
                    ]
                )

        # System-wide optimizations
        lines.extend(
            [
                "# =============================================",
                "# SYSTEM-WIDE OPTIMIZATIONS",
                "# =============================================",
                "",
                'echo ""',
                'echo "Applying system-wide optimizations..."',
                "",
                "# Increase file descriptor limits",
                'ulimit -n 65535 2>/dev/null || echo "  ⚠ Could not increase file descriptor limit"',
                "",
                "# Enable TCP keepalive tuning",
                "sudo sysctl -w net.ipv4.tcp_keepalive_time=60 2>/dev/null",
                "sudo sysctl -w net.ipv4.tcp_keepalive_intvl=10 2>/dev/null",
                "sudo sysctl -w net.ipv4.tcp_keepalive_probes=6 2>/dev/null",
                'echo "  ✓ TCP keepalive tuned"',
                "",
                "# Optimize virtual memory",
                "sudo sysctl -w vm.dirty_ratio=10 2>/dev/null",
                "sudo sysctl -w vm.dirty_background_ratio=5 2>/dev/null",
                'echo "  ✓ Virtual memory settings optimized"',
                "",
                'echo ""',
                'echo "=========================================="',
                'echo "  Optimization Complete!"',
                'echo "=========================================="',
                'echo "  Re-run strace analysis to measure improvement."',
                "",
            ]
        )

        content = "\n".join(lines)

        with open(filepath, "w") as f:
            f.write(content)

        # Make executable
        os.chmod(filepath, 0o755)

        return {
            "success": True,
            "filepath": filepath,
            "filename": filename,
            "content": content,
            "size": len(content),
        }

    def export_json_report(
        self, recommendations, analysis=None, strace_stats=None, filename=None
    ):
        """Export a JSON report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "SysCall AI Platform",
                "version": "1.0",
            },
            "analysis": analysis or {},
            "recommendations": recommendations,
            "strace_parse_stats": strace_stats or {},
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return {
            "success": True,
            "filepath": filepath,
            "filename": filename,
            "content": json.dumps(report, indent=2, default=str),
        }

    def export_markdown_report(
        self, recommendations, analysis=None, strace_stats=None, filename=None
    ):
        """Export a Markdown report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"

        filepath = os.path.join(self.output_dir, filename)

        lines = [
            "# SysCall AI - Optimization Report",
            f"",
            f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            "",
        ]

        # Strace parse stats
        if strace_stats:
            lines.extend(
                [
                    "## Strace Parse Statistics",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f'| Parsed Lines | {strace_stats.get("parsed_lines", "N/A")} |',
                    f'| Skipped Lines | {strace_stats.get("skipped_lines", "N/A")} |',
                    f'| Parse Errors | {strace_stats.get("total_errors", "N/A")} |',
                    "",
                ]
            )

        # Analysis summary
        if analysis:
            lines.extend(
                [
                    "## Analysis Summary",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f'| Total Syscalls | {analysis.get("total_syscalls", "N/A")} |',
                    f'| High Latency Calls | {analysis.get("high_latency_count", "N/A")} |',
                    f'| Optimization Candidates | {analysis.get("optimization_candidates", "N/A")} |',
                    f'| Anomalies Detected | {analysis.get("anomalies_detected", "N/A")} |',
                    f'| Avg Latency | {analysis.get("avg_latency", "N/A"):.2f} μs |'
                    if isinstance(analysis.get("avg_latency"), (int, float))
                    else f"| Avg Latency | N/A |",
                    f'| P95 Latency | {analysis.get("p95_latency", "N/A"):.2f} μs |'
                    if isinstance(analysis.get("p95_latency"), (int, float))
                    else f"| P95 Latency | N/A |",
                    "",
                ]
            )

        # Recommendations
        if recommendations:
            lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )

            for i, rec in enumerate(recommendations, 1):
                severity = rec.get("severity", "info")
                severity_icon = (
                    "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                )

                lines.extend(
                    [
                        f'### {i}. {severity_icon} {rec.get("type", "General").replace("_", " ").title()}',
                        "",
                        f"- **Severity:** {severity}",
                    ]
                )

                if "syscall" in rec:
                    lines.append(f'- **Syscall:** `{rec["syscall"]}`')
                if "count" in rec:
                    lines.append(f'- **Affected Calls:** {rec["count"]}')
                if "avg_latency" in rec:
                    lines.append(f'- **Avg Latency:** {rec["avg_latency"]:.2f} μs')

                lines.extend(
                    [
                        f'- **Action:** {rec.get("recommendation", "N/A")}',
                        "",
                    ]
                )

        content = "\n".join(lines)

        with open(filepath, "w") as f:
            f.write(content)

        return {
            "success": True,
            "filepath": filepath,
            "filename": filename,
            "content": content,
        }

    def list_exports(self):
        """List all exported files"""
        exports = []
        if os.path.exists(self.output_dir):
            for f in sorted(os.listdir(self.output_dir)):
                filepath = os.path.join(self.output_dir, f)
                exports.append(
                    {
                        "filename": f,
                        "filepath": filepath,
                        "size": os.path.getsize(filepath),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        ).isoformat(),
                    }
                )
        return exports
