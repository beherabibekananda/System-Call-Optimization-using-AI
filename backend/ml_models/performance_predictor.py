"""
ML-based Performance Predictor
Predicts process scheduling and resource allocation optimizations
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import os


class PerformancePredictor:
    """
    Predicts system performance metrics and suggests resource allocation
    """

    def __init__(self):
        self.latency_predictor = None
        self.throughput_predictor = None
        self.resource_optimizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.training_metrics = {}

    def prepare_process_features(self, df, fit=False):
        """Prepare process-level features for prediction"""
        # Aggregate syscall data to process level
        if "pid" not in df.columns:
            df["pid"] = 0

        process_stats = (
            df.groupby("pid")
            .agg(
                {
                    "latency_us": ["mean", "std", "sum", "max", "min", "count"],
                    "cpu_usage": ["mean", "max"],
                    "memory_usage": ["mean", "max"],
                    "context_switches": ["sum", "mean"],
                    "is_high_latency": "sum",
                    "is_redundant": "sum",
                    "risk_level": "mean",
                }
            )
            .reset_index()
        )

        # Flatten column names
        process_stats.columns = [
            "pid",
            "lat_mean",
            "lat_std",
            "lat_sum",
            "lat_max",
            "lat_min",
            "syscall_count",
            "cpu_mean",
            "cpu_max",
            "mem_mean",
            "mem_max",
            "ctx_sum",
            "ctx_mean",
            "high_lat_count",
            "redundant_count",
            "risk_mean",
        ]

        # Add process type if available
        if "process_type" in df.columns:
            process_types = df.groupby("pid")["process_type"].first().reset_index()
            process_stats = process_stats.merge(process_types, on="pid")

            if fit:
                self.label_encoders["process_type"] = LabelEncoder()
                process_stats["process_type_encoded"] = self.label_encoders[
                    "process_type"
                ].fit_transform(process_stats["process_type"])
            else:
                process_stats["process_type_encoded"] = self.label_encoders[
                    "process_type"
                ].transform(process_stats["process_type"])

        # Calculate derived metrics
        process_stats["efficiency"] = process_stats["syscall_count"] / (
            process_stats["lat_sum"] + 1
        )
        process_stats["high_lat_ratio"] = (
            process_stats["high_lat_count"] / process_stats["syscall_count"]
        )
        process_stats["redundancy_ratio"] = (
            process_stats["redundant_count"] / process_stats["syscall_count"]
        )
        process_stats["latency_cv"] = process_stats["lat_std"] / (
            process_stats["lat_mean"] + 0.01
        )

        # Target metrics (simulated for training)
        np.random.seed(42)
        process_stats["predicted_throughput"] = (
            1000
            / (process_stats["lat_mean"] + 1)
            * (1 - process_stats["high_lat_ratio"])
            * np.random.uniform(0.8, 1.2, len(process_stats))
        )

        process_stats["optimal_cpu_allocation"] = np.clip(
            process_stats["cpu_mean"] * (1 + process_stats["high_lat_ratio"]), 10, 100
        )

        process_stats["optimal_memory_allocation"] = np.clip(
            process_stats["mem_mean"] * (1 + process_stats["redundancy_ratio"] * 0.5),
            10,
            100,
        )

        return process_stats

    def train(self, df, test_size=0.2):
        """Train performance prediction models"""
        print("Preparing process-level features...")
        process_data = self.prepare_process_features(df, fit=True)

        # Feature columns
        feature_cols = [
            "lat_mean",
            "lat_std",
            "lat_max",
            "syscall_count",
            "cpu_mean",
            "cpu_max",
            "mem_mean",
            "mem_max",
            "ctx_sum",
            "high_lat_count",
            "redundant_count",
            "risk_mean",
            "efficiency",
            "high_lat_ratio",
            "redundancy_ratio",
            "latency_cv",
        ]

        if "process_type_encoded" in process_data.columns:
            feature_cols.append("process_type_encoded")

        X = process_data[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Targets
        y_throughput = process_data["predicted_throughput"]
        y_cpu = process_data["optimal_cpu_allocation"]
        y_memory = process_data["optimal_memory_allocation"]

        # Split data
        X_train, X_test, y_thr_train, y_thr_test = train_test_split(
            X_scaled, y_throughput, test_size=test_size, random_state=42
        )
        _, _, y_cpu_train, y_cpu_test = train_test_split(
            X_scaled, y_cpu, test_size=test_size, random_state=42
        )
        _, _, y_mem_train, y_mem_test = train_test_split(
            X_scaled, y_memory, test_size=test_size, random_state=42
        )

        # Train Throughput Predictor
        print("\nTraining Throughput Predictor...")
        self.throughput_predictor = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.throughput_predictor.fit(X_train, y_thr_train)

        thr_pred = self.throughput_predictor.predict(X_test)
        thr_r2 = r2_score(y_thr_test, thr_pred)
        thr_mae = mean_absolute_error(y_thr_test, thr_pred)
        print(f"Throughput Prediction - R²: {thr_r2:.4f}, MAE: {thr_mae:.4f}")

        # Train Resource Optimizer (for CPU and Memory)
        print("\nTraining Resource Optimizer...")
        self.resource_optimizer = {
            "cpu": GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42
            ),
            "memory": GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42
            ),
        }

        self.resource_optimizer["cpu"].fit(X_train, y_cpu_train)
        self.resource_optimizer["memory"].fit(X_train, y_mem_train)

        cpu_pred = self.resource_optimizer["cpu"].predict(X_test)
        mem_pred = self.resource_optimizer["memory"].predict(X_test)

        cpu_r2 = r2_score(y_cpu_test, cpu_pred)
        mem_r2 = r2_score(y_mem_test, mem_pred)

        print(f"CPU Optimization - R²: {cpu_r2:.4f}")
        print(f"Memory Optimization - R²: {mem_r2:.4f}")

        # Store metrics
        self.training_metrics = {
            "timestamp": datetime.now().isoformat(),
            "samples_trained": len(X_train),
            "throughput_model": {"r2_score": float(thr_r2), "mae": float(thr_mae)},
            "resource_optimizer": {"cpu_r2": float(cpu_r2), "memory_r2": float(mem_r2)},
            "feature_columns": feature_cols,
        }

        self.feature_columns = feature_cols
        self.is_trained = True
        print("\n✓ Performance predictor trained successfully!")

        return self.training_metrics

    def predict_performance(self, df):
        """Predict performance metrics for processes"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")

        process_data = self.prepare_process_features(df, fit=False)

        # Prepare features
        feature_cols = [c for c in self.feature_columns if c in process_data.columns]
        X = process_data[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Predictions
        process_data["predicted_throughput"] = self.throughput_predictor.predict(
            X_scaled
        )
        process_data["recommended_cpu"] = self.resource_optimizer["cpu"].predict(
            X_scaled
        )
        process_data["recommended_memory"] = self.resource_optimizer["memory"].predict(
            X_scaled
        )

        # Calculate potential improvements
        process_data["cpu_adjustment"] = (
            process_data["recommended_cpu"] - process_data["cpu_mean"]
        )
        process_data["memory_adjustment"] = (
            process_data["recommended_memory"] - process_data["mem_mean"]
        )

        # Priority score
        process_data["optimization_priority"] = (
            abs(process_data["cpu_adjustment"]) * 0.3
            + abs(process_data["memory_adjustment"]) * 0.3
            + process_data["high_lat_ratio"] * 100 * 0.4
        )

        return process_data

    def get_scheduling_recommendations(self, df):
        """Generate process scheduling recommendations"""
        predictions = self.predict_performance(df)

        recommendations = []

        for _, proc in predictions.iterrows():
            rec = {
                "pid": int(proc["pid"]),
                "current_cpu": float(proc["cpu_mean"]),
                "current_memory": float(proc["mem_mean"]),
                "recommended_cpu": float(proc["recommended_cpu"]),
                "recommended_memory": float(proc["recommended_memory"]),
                "predicted_throughput": float(proc["predicted_throughput"]),
                "priority_score": float(proc["optimization_priority"]),
                "actions": [],
            }

            # CPU recommendations
            if proc["cpu_adjustment"] > 10:
                rec["actions"].append(
                    {
                        "type": "increase_cpu",
                        "value": float(proc["cpu_adjustment"]),
                        "reason": "Process is CPU-constrained with high latency calls",
                    }
                )
            elif proc["cpu_adjustment"] < -10:
                rec["actions"].append(
                    {
                        "type": "decrease_cpu",
                        "value": float(abs(proc["cpu_adjustment"])),
                        "reason": "Process is over-allocated on CPU resources",
                    }
                )

            # Memory recommendations
            if proc["memory_adjustment"] > 10:
                rec["actions"].append(
                    {
                        "type": "increase_memory",
                        "value": float(proc["memory_adjustment"]),
                        "reason": "Process has high redundancy, may benefit from caching",
                    }
                )
            elif proc["memory_adjustment"] < -10:
                rec["actions"].append(
                    {
                        "type": "decrease_memory",
                        "value": float(abs(proc["memory_adjustment"])),
                        "reason": "Process is over-allocated on memory resources",
                    }
                )

            # Scheduling recommendations
            if proc["high_lat_ratio"] > 0.2:
                rec["actions"].append(
                    {
                        "type": "priority_boost",
                        "reason": "High latency ratio suggests process may benefit from higher priority",
                    }
                )

            if proc["redundancy_ratio"] > 0.3:
                rec["actions"].append(
                    {
                        "type": "code_optimization",
                        "reason": "High redundancy suggests code-level optimization opportunities",
                    }
                )

            recommendations.append(rec)

        # Sort by priority
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)

        return recommendations

    def save_models(self, path="models"):
        """Save trained models"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")

        os.makedirs(path, exist_ok=True)

        joblib.dump(self.throughput_predictor, f"{path}/throughput_predictor.pkl")
        joblib.dump(self.resource_optimizer, f"{path}/resource_optimizer.pkl")
        joblib.dump(self.scaler, f"{path}/perf_scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/perf_label_encoders.pkl")

        with open(f"{path}/perf_config.json", "w") as f:
            json.dump(
                {
                    "feature_columns": self.feature_columns,
                    "training_metrics": self.training_metrics,
                },
                f,
                indent=2,
            )

        print(f"Performance predictor saved to {path}/")

    def load_models(self, path="models"):
        """Load trained models"""
        self.throughput_predictor = joblib.load(f"{path}/throughput_predictor.pkl")
        self.resource_optimizer = joblib.load(f"{path}/resource_optimizer.pkl")
        self.scaler = joblib.load(f"{path}/perf_scaler.pkl")
        self.label_encoders = joblib.load(f"{path}/perf_label_encoders.pkl")

        with open(f"{path}/perf_config.json", "r") as f:
            config = json.load(f)
            self.feature_columns = config["feature_columns"]
            self.training_metrics = config["training_metrics"]

        self.is_trained = True
        print(f"Performance predictor loaded from {path}/")


def main():
    """Example usage"""
    import sys

    sys.path.append("..")
    from data.syscall_data_generator import SyscallDataGenerator

    # Generate data
    generator = SyscallDataGenerator()
    benchmark_data = generator.generate_benchmark_data(
        num_processes=50, calls_per_process=500
    )

    # Train predictor
    predictor = PerformancePredictor()
    metrics = predictor.train(benchmark_data)

    print("\n=== Training Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Get recommendations
    recommendations = predictor.get_scheduling_recommendations(benchmark_data)

    print("\n=== Top 5 Scheduling Recommendations ===")
    for rec in recommendations[:5]:
        print(f"\nPID {rec['pid']} (Priority: {rec['priority_score']:.2f})")
        for action in rec["actions"]:
            print(f"  - {action['type']}: {action['reason']}")


if __name__ == "__main__":
    main()
