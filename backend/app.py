"""
Flask API for System Call Optimization Platform
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models.syscall_analyzer import SyscallAnalyzer
from ml_models.performance_predictor import PerformancePredictor
from data.syscall_data_generator import SyscallDataGenerator, SYSCALL_CATALOG
from strace.strace_parser import StraceParser
from strace.strace_runner import StraceRunner
from strace.script_exporter import ScriptExporter

app = Flask(
    __name__,
    static_folder="../frontend/static",
    template_folder="../frontend/templates",
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global instances
analyzer = SyscallAnalyzer()
predictor = PerformancePredictor()
data_generator = SyscallDataGenerator()
strace_parser = StraceParser()
strace_runner = StraceRunner()
script_exporter = ScriptExporter()

# Store for real-time data
realtime_data = {"syscalls": [], "metrics": {}, "is_monitoring": False}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_models")
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "strace", "samples")
UPLOAD_PATH = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_PATH, exist_ok=True)


def ensure_models_trained():
    """Ensure models are trained with initial data"""
    global analyzer, predictor

    if not analyzer.is_trained:
        print("Training models with synthetic data...")
        training_data = data_generator.generate_training_dataset(samples_per_type=2000)
        analyzer.train(training_data)

        benchmark_data = data_generator.generate_benchmark_data(
            num_processes=20, calls_per_process=200
        )
        predictor.train(benchmark_data)

        # Save models
        os.makedirs(MODEL_PATH, exist_ok=True)
        analyzer.save_models(MODEL_PATH)
        predictor.save_models(MODEL_PATH)
        print("Models trained and saved!")


# Routes
@app.route("/")
def index():
    """Serve the main dashboard"""
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    """Get system status"""
    return jsonify(
        {
            "status": "ok",
            "models_trained": analyzer.is_trained,
            "timestamp": datetime.now().isoformat(),
            "available_syscalls": len(SYSCALL_CATALOG),
            "is_monitoring": realtime_data["is_monitoring"],
        }
    )


@app.route("/api/train", methods=["POST"])
def train_models():
    """Train or retrain models"""
    try:
        samples = request.json.get("samples_per_type", 2000)

        training_data = data_generator.generate_training_dataset(
            samples_per_type=samples
        )
        analyzer_metrics = analyzer.train(training_data)

        benchmark_data = data_generator.generate_benchmark_data(
            num_processes=30, calls_per_process=300
        )
        predictor_metrics = predictor.train(benchmark_data)

        # Save models
        os.makedirs(MODEL_PATH, exist_ok=True)
        analyzer.save_models(MODEL_PATH)
        predictor.save_models(MODEL_PATH)

        return jsonify(
            {
                "success": True,
                "analyzer_metrics": analyzer_metrics,
                "predictor_metrics": predictor_metrics,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_syscalls():
    """Analyze syscall data"""
    try:
        ensure_models_trained()

        data = request.json

        if "generate" in data:
            # Generate synthetic data for testing
            num_calls = data.get("num_calls", 1000)
            process_type = data.get("process_type", "mixed")
            df = data_generator.generate_syscall_trace(num_calls, process_type)
        else:
            # Use provided data
            df = pd.DataFrame(data.get("syscalls", []))

        recommendations, analysis = analyzer.get_optimization_recommendations(df)

        return jsonify(
            {"success": True, "analysis": analysis, "recommendations": recommendations}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_performance():
    """Predict performance and get scheduling recommendations"""
    try:
        ensure_models_trained()

        data = request.json

        if "generate" in data:
            num_processes = data.get("num_processes", 10)
            calls_per_process = data.get("calls_per_process", 200)
            df = data_generator.generate_benchmark_data(
                num_processes, calls_per_process
            )
        else:
            df = pd.DataFrame(data.get("syscalls", []))

        recommendations = predictor.get_scheduling_recommendations(df)

        return jsonify({"success": True, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """Run performance benchmark"""
    try:
        ensure_models_trained()

        data = request.json
        num_iterations = data.get("iterations", 5)
        calls_per_iteration = data.get("calls_per_iteration", 500)

        results = []

        for i in range(num_iterations):
            # Generate data
            df = data_generator.generate_syscall_trace(calls_per_iteration, "mixed")

            # Before optimization metrics
            before_metrics = {
                "avg_latency": float(df["latency_us"].mean()),
                "p95_latency": float(df["latency_us"].quantile(0.95)),
                "high_latency_count": int(df["is_high_latency"].sum()),
                "redundant_count": int(df["is_redundant"].sum()),
                "cpu_usage": float(df["cpu_usage"].mean()),
                "memory_usage": float(df["memory_usage"].mean()),
            }

            # Get optimization predictions
            _, analysis = analyzer.analyze_patterns(df)

            # Simulate "after optimization" with reduced latency
            optimization_factor = 1 - (analysis["high_latency_count"] / len(df) * 0.3)

            after_metrics = {
                "avg_latency": float(
                    before_metrics["avg_latency"] * optimization_factor
                ),
                "p95_latency": float(
                    before_metrics["p95_latency"] * optimization_factor
                ),
                "high_latency_count": int(before_metrics["high_latency_count"] * 0.3),
                "redundant_count": int(before_metrics["redundant_count"] * 0.4),
                "cpu_usage": float(before_metrics["cpu_usage"] * 0.85),
                "memory_usage": float(before_metrics["memory_usage"] * 0.9),
            }

            improvement = {
                "latency_reduction": float((1 - optimization_factor) * 100),
                "cpu_savings": float(
                    (before_metrics["cpu_usage"] - after_metrics["cpu_usage"])
                    / before_metrics["cpu_usage"]
                    * 100
                ),
                "memory_savings": float(
                    (before_metrics["memory_usage"] - after_metrics["memory_usage"])
                    / before_metrics["memory_usage"]
                    * 100
                ),
                "throughput_increase": float((1 / optimization_factor - 1) * 100),
            }

            results.append(
                {
                    "iteration": i + 1,
                    "before": before_metrics,
                    "after": after_metrics,
                    "improvement": improvement,
                }
            )

        # Aggregate results
        avg_improvement = {
            "latency_reduction": np.mean(
                [r["improvement"]["latency_reduction"] for r in results]
            ),
            "cpu_savings": np.mean([r["improvement"]["cpu_savings"] for r in results]),
            "memory_savings": np.mean(
                [r["improvement"]["memory_savings"] for r in results]
            ),
            "throughput_increase": np.mean(
                [r["improvement"]["throughput_increase"] for r in results]
            ),
        }

        return jsonify(
            {
                "success": True,
                "iterations": results,
                "average_improvement": avg_improvement,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/syscall-catalog")
def get_syscall_catalog():
    """Get the syscall catalog"""
    return jsonify({"success": True, "catalog": SYSCALL_CATALOG})


@app.route("/api/model-metrics")
def get_model_metrics():
    """Get current model metrics"""
    try:
        ensure_models_trained()

        return jsonify(
            {
                "success": True,
                "analyzer_metrics": analyzer.training_metrics,
                "predictor_metrics": predictor.training_metrics,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================
# Strace Integration Endpoints
# ============================================


@app.route("/api/strace/parse", methods=["POST"])
def parse_strace_file():
    """Parse an uploaded strace output file"""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        # Save uploaded file
        filepath = os.path.join(UPLOAD_PATH, file.filename)
        file.save(filepath)

        # Parse the file
        df = strace_parser.parse_file(filepath)
        parse_stats = strace_parser.get_parse_stats()

        if df.empty:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No syscalls could be parsed from the file",
                        "parse_stats": parse_stats,
                    }
                ),
                400,
            )

        # Run ML analysis
        ensure_models_trained()
        recommendations, analysis = analyzer.get_optimization_recommendations(df)

        return jsonify(
            {
                "success": True,
                "parse_stats": parse_stats,
                "total_syscalls": len(df),
                "analysis": analysis,
                "recommendations": recommendations,
                "syscall_summary": df["syscall_name"].value_counts().head(15).to_dict(),
                "category_summary": df["category"].value_counts().to_dict(),
                "source": "strace_file",
                "filename": file.filename,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/strace/parse-text", methods=["POST"])
def parse_strace_text():
    """Parse pasted strace text output"""
    try:
        data = request.json
        strace_text = data.get("text", "")

        if not strace_text.strip():
            return jsonify({"success": False, "error": "No strace text provided"}), 400

        # Parse the text
        df = strace_parser.parse_text(strace_text)
        parse_stats = strace_parser.get_parse_stats()

        if df.empty:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No syscalls could be parsed from the text",
                        "parse_stats": parse_stats,
                    }
                ),
                400,
            )

        # Run ML analysis
        ensure_models_trained()
        recommendations, analysis = analyzer.get_optimization_recommendations(df)

        return jsonify(
            {
                "success": True,
                "parse_stats": parse_stats,
                "total_syscalls": len(df),
                "analysis": analysis,
                "recommendations": recommendations,
                "syscall_summary": df["syscall_name"].value_counts().head(15).to_dict(),
                "category_summary": df["category"].value_counts().to_dict(),
                "source": "strace_text",
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/strace/run", methods=["POST"])
def run_strace():
    """Run strace on a command (Linux) or dtruss (macOS)"""
    try:
        data = request.json
        command = data.get("command", "")
        timeout = data.get("timeout", 30)

        if not command:
            return jsonify({"success": False, "error": "No command provided"}), 400

        if not strace_runner.is_available():
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No tracing tool available",
                        "tool_info": strace_runner.get_tool_info(),
                    }
                ),
                400,
            )

        result = strace_runner.trace_command(command, timeout=timeout)

        if not result["success"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": result["error"],
                        "tool_info": strace_runner.get_tool_info(),
                    }
                ),
                400,
            )

        df = result["data"]

        if df is not None and not df.empty:
            # Run ML analysis
            ensure_models_trained()
            recommendations, analysis = analyzer.get_optimization_recommendations(df)

            return jsonify(
                {
                    "success": True,
                    "stats": result["stats"],
                    "total_syscalls": len(df),
                    "analysis": analysis,
                    "recommendations": recommendations,
                    "syscall_summary": df["syscall_name"]
                    .value_counts()
                    .head(15)
                    .to_dict(),
                    "category_summary": df["category"].value_counts().to_dict(),
                    "source": "strace_live",
                    "command": command,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No syscalls captured",
                        "stats": result["stats"],
                    }
                ),
                400,
            )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/strace/tool-info")
def strace_tool_info():
    """Get info about available tracing tools"""
    return jsonify({"success": True, "tool_info": strace_runner.get_tool_info()})


@app.route("/api/strace/samples")
def list_strace_samples():
    """List available sample strace files"""
    samples = []
    if os.path.exists(SAMPLES_PATH):
        for f in os.listdir(SAMPLES_PATH):
            if f.endswith(".strace"):
                filepath = os.path.join(SAMPLES_PATH, f)
                samples.append(
                    {
                        "filename": f,
                        "size": os.path.getsize(filepath),
                        "description": _get_sample_description(f),
                    }
                )
    return jsonify({"success": True, "samples": samples})


@app.route("/api/strace/load-sample", methods=["POST"])
def load_strace_sample():
    """Load and parse a sample strace file"""
    try:
        data = request.json
        filename = data.get("filename", "")

        filepath = os.path.join(SAMPLES_PATH, filename)
        if not os.path.exists(filepath):
            return (
                jsonify({"success": False, "error": f"Sample not found: {filename}"}),
                404,
            )

        # Parse the sample
        df = strace_parser.parse_file(filepath)
        parse_stats = strace_parser.get_parse_stats()

        if df.empty:
            return jsonify({"success": False, "error": "No syscalls parsed"}), 400

        # Run ML analysis
        ensure_models_trained()
        recommendations, analysis = analyzer.get_optimization_recommendations(df)

        return jsonify(
            {
                "success": True,
                "parse_stats": parse_stats,
                "total_syscalls": len(df),
                "analysis": analysis,
                "recommendations": recommendations,
                "syscall_summary": df["syscall_name"].value_counts().head(15).to_dict(),
                "category_summary": df["category"].value_counts().to_dict(),
                "source": "strace_sample",
                "filename": filename,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _get_sample_description(filename):
    """Get description for a sample strace file"""
    descriptions = {
        "sample_ls.strace": "Trace of ls -la /tmp command (file operations)",
        "sample_webserver.strace": "Web server handling HTTP requests (network + I/O)",
        "sample_summary.strace": "Strace -c summary output (aggregated stats)",
    }
    return descriptions.get(filename, "Strace sample output")


# ============================================
# Export Endpoints
# ============================================


@app.route("/api/export", methods=["POST"])
def export_report():
    """Export recommendations as shell script, JSON, or Markdown"""
    try:
        data = request.json
        export_format = data.get("format", "script")  # script, json, markdown
        recommendations = data.get("recommendations", [])
        analysis = data.get("analysis", {})
        strace_stats = data.get("strace_stats", {})

        # If no recommendations provided, generate fresh ones
        if not recommendations:
            ensure_models_trained()
            test_data = data_generator.generate_syscall_trace(1000, "mixed")
            recommendations, analysis = analyzer.get_optimization_recommendations(
                test_data
            )

        if export_format == "script":
            result = script_exporter.export_optimization_script(
                recommendations, analysis
            )
        elif export_format == "json":
            result = script_exporter.export_json_report(
                recommendations, analysis, strace_stats
            )
        elif export_format == "markdown":
            result = script_exporter.export_markdown_report(
                recommendations, analysis, strace_stats
            )
        else:
            return (
                jsonify(
                    {"success": False, "error": f"Unknown format: {export_format}"}
                ),
                400,
            )

        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/exports")
def list_exports():
    """List all exported files"""
    return jsonify({"success": True, "exports": script_exporter.list_exports()})


# Real-time monitoring via WebSocket
def monitor_syscalls():
    """Simulate real-time syscall monitoring"""
    while realtime_data["is_monitoring"]:
        # Generate a batch of syscalls
        df = data_generator.generate_syscall_trace(10, "mixed")

        for _, row in df.iterrows():
            if not realtime_data["is_monitoring"]:
                break

            syscall_event = {
                "timestamp": datetime.now().isoformat(),
                "syscall": row["syscall_name"],
                "category": row["category"],
                "latency": float(row["latency_us"]),
                "cpu": float(row["cpu_usage"]),
                "memory": float(row["memory_usage"]),
                "is_high_latency": bool(row["is_high_latency"]),
                "is_redundant": bool(row["is_redundant"]),
            }

            socketio.emit("syscall_event", syscall_event)
            time.sleep(0.1)  # 100ms between events


@socketio.on("start_monitoring")
def handle_start_monitoring():
    """Start real-time monitoring"""
    if not realtime_data["is_monitoring"]:
        realtime_data["is_monitoring"] = True
        thread = threading.Thread(target=monitor_syscalls)
        thread.daemon = True
        thread.start()
        emit("monitoring_status", {"status": "started"})


@socketio.on("stop_monitoring")
def handle_stop_monitoring():
    """Stop real-time monitoring"""
    realtime_data["is_monitoring"] = False
    emit("monitoring_status", {"status": "stopped"})


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    emit("connection_status", {"status": "connected"})


if __name__ == "__main__":
    print("=" * 50)
    print("System Call Optimization Platform")
    print("=" * 50)

    # Train models on startup
    ensure_models_trained()

    print("\nStarting server on http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
