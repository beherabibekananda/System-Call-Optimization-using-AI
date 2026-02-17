"""
AI-based System Call Analyzer
Uses machine learning to identify high-latency and redundant kernel calls
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
import joblib
import json
from datetime import datetime
import os


class SyscallAnalyzer:
    """
    ML-based analyzer for system call optimization
    Identifies high-latency calls, redundancy patterns, and optimization opportunities
    """

    def __init__(self):
        self.latency_model = None
        self.redundancy_model = None
        self.anomaly_detector = None
        self.pattern_clusterer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.training_metrics = {}

    def prepare_features(self, df, fit=False):
        """Prepare features for ML models"""
        df = df.copy()

        # Encode categorical variables
        categorical_cols = (
            ["syscall_name", "category", "process_type"]
            if "process_type" in df.columns
            else ["syscall_name", "category"]
        )

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        known_labels = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(
                            lambda x: x if x in known_labels else "unknown"
                        )
                        if "unknown" not in self.label_encoders[col].classes_:
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, "unknown"
                            )
                        df[f"{col}_encoded"] = self.label_encoders[col].transform(
                            df[col].astype(str)
                        )

        # Create derived features
        df["latency_log"] = np.log1p(df["latency_us"])
        df["latency_squared"] = df["latency_us"] ** 2
        df["cpu_memory_ratio"] = df["cpu_usage"] / (df["memory_usage"] + 0.1)
        df["efficiency_score"] = df["base_latency_us"] / (df["latency_us"] + 0.1)

        # Feature columns for modeling
        feature_cols = [
            "syscall_name_encoded",
            "category_encoded",
            "latency_us",
            "base_latency_us",
            "latency_ratio",
            "latency_log",
            "cpu_usage",
            "memory_usage",
            "cpu_memory_ratio",
            "context_switches",
            "risk_level",
            "efficiency_score",
        ]

        if "process_type_encoded" in df.columns:
            feature_cols.append("process_type_encoded")

        self.feature_columns = feature_cols

        X = df[feature_cols].fillna(0)

        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, df

    def train(self, df, test_size=0.2):
        """Train all ML models on syscall data"""
        print("Preparing features...")
        X, df = self.prepare_features(df, fit=True)

        # Target variables
        y_high_latency = df["is_high_latency"].astype(int)
        y_needs_optimization = df["needs_optimization"].astype(int)

        # Split data
        X_train, X_test, y_lat_train, y_lat_test = train_test_split(
            X, y_high_latency, test_size=test_size, random_state=42
        )
        _, _, y_opt_train, y_opt_test = train_test_split(
            X, y_needs_optimization, test_size=test_size, random_state=42
        )

        # Train High Latency Detector (Random Forest)
        print("\nTraining High Latency Detector...")
        self.latency_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.latency_model.fit(X_train, y_lat_train)

        lat_pred = self.latency_model.predict(X_test)
        lat_proba = self.latency_model.predict_proba(X_test)[:, 1]

        print("High Latency Detection Results:")
        print(classification_report(y_lat_test, lat_pred))
        lat_auc = roc_auc_score(y_lat_test, lat_proba)
        print(f"AUC-ROC: {lat_auc:.4f}")

        # Train Optimization Recommender (Random Forest)
        print("\nTraining Optimization Recommender...")
        self.redundancy_model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        )
        self.redundancy_model.fit(X_train, y_opt_train)

        opt_pred = self.redundancy_model.predict(X_test)
        opt_proba = self.redundancy_model.predict_proba(X_test)[:, 1]

        print("Optimization Recommendation Results:")
        print(classification_report(y_opt_test, opt_pred))
        opt_auc = roc_auc_score(y_opt_test, opt_proba)
        print(f"AUC-ROC: {opt_auc:.4f}")

        # Train Anomaly Detector (Isolation Forest)
        print("\nTraining Anomaly Detector...")
        self.anomaly_detector = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=42
        )
        self.anomaly_detector.fit(X_train)

        # Train Pattern Clusterer (KMeans)
        print("\nTraining Pattern Clusterer...")
        self.pattern_clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.pattern_clusterer.fit(X_train)

        # Store training metrics
        self.training_metrics = {
            "timestamp": datetime.now().isoformat(),
            "samples_trained": len(X_train),
            "samples_tested": len(X_test),
            "latency_model": {
                "auc_roc": float(lat_auc),
                "accuracy": float((lat_pred == y_lat_test).mean()),
            },
            "optimization_model": {
                "auc_roc": float(opt_auc),
                "accuracy": float((opt_pred == y_opt_test).mean()),
            },
            "feature_importance": dict(
                zip(
                    self.feature_columns,
                    self.latency_model.feature_importances_.tolist(),
                )
            ),
        }

        self.is_trained = True
        print("\n✓ All models trained successfully!")

        return self.training_metrics

    def predict(self, df):
        """Make predictions on new syscall data"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")

        X, df = self.prepare_features(df, fit=False)

        # Predictions
        high_latency_prob = self.latency_model.predict_proba(X)[:, 1]
        optimization_prob = self.redundancy_model.predict_proba(X)[:, 1]
        anomaly_scores = self.anomaly_detector.decision_function(X)
        pattern_clusters = self.pattern_clusterer.predict(X)

        # Add predictions to dataframe
        results = df.copy()
        results["high_latency_probability"] = high_latency_prob
        results["optimization_probability"] = optimization_prob
        results["anomaly_score"] = anomaly_scores
        results["pattern_cluster"] = pattern_clusters
        results["is_anomaly"] = anomaly_scores < 0

        # Calculate optimization priority
        results["optimization_priority"] = (
            results["high_latency_probability"] * 0.4
            + results["optimization_probability"] * 0.4
            + (1 - (anomaly_scores + 0.5).clip(0, 1)) * 0.2
        )

        return results

    def analyze_patterns(self, df):
        """Analyze syscall patterns and generate insights"""
        results = self.predict(df)

        analysis = {
            "total_syscalls": len(results),
            "high_latency_count": int(
                (results["high_latency_probability"] > 0.5).sum()
            ),
            "optimization_candidates": int(
                (results["optimization_probability"] > 0.5).sum()
            ),
            "anomalies_detected": int(results["is_anomaly"].sum()),
            "avg_latency": float(results["latency_us"].mean()),
            "p95_latency": float(results["latency_us"].quantile(0.95)),
            "p99_latency": float(results["latency_us"].quantile(0.99)),
            "cpu_avg": float(results["cpu_usage"].mean()),
            "memory_avg": float(results["memory_usage"].mean()),
        }

        # Top syscalls needing optimization
        top_optimization = (
            results.groupby("syscall_name")
            .agg(
                {
                    "optimization_priority": "mean",
                    "latency_us": "mean",
                    "syscall_id": "count",
                }
            )
            .rename(columns={"syscall_id": "count"})
            .sort_values("optimization_priority", ascending=False)
            .head(10)
        )

        analysis["top_optimization_targets"] = top_optimization.to_dict("index")

        # Pattern analysis by cluster
        cluster_analysis = (
            results.groupby("pattern_cluster")
            .agg(
                {
                    "syscall_name": lambda x: x.mode().iloc[0]
                    if len(x) > 0
                    else "unknown",
                    "latency_us": "mean",
                    "optimization_priority": "mean",
                    "syscall_id": "count",
                }
            )
            .rename(columns={"syscall_id": "count", "syscall_name": "dominant_syscall"})
        )

        analysis["pattern_clusters"] = cluster_analysis.to_dict("index")

        # Category breakdown
        category_stats = results.groupby("category").agg(
            {
                "latency_us": ["mean", "sum", "count"],
                "high_latency_probability": "mean",
                "optimization_probability": "mean",
            }
        )
        category_stats.columns = [
            "avg_latency",
            "total_latency",
            "count",
            "high_lat_prob",
            "opt_prob",
        ]
        analysis["category_breakdown"] = category_stats.to_dict("index")

        return analysis, results

    def get_optimization_recommendations(self, df):
        """Generate specific optimization recommendations"""
        analysis, results = self.analyze_patterns(df)

        recommendations = []

        # High latency syscalls
        high_lat = results[results["high_latency_probability"] > 0.7]
        if len(high_lat) > 0:
            for syscall in high_lat["syscall_name"].unique()[:5]:
                syscall_data = high_lat[high_lat["syscall_name"] == syscall]
                recommendations.append(
                    {
                        "type": "high_latency",
                        "syscall": syscall,
                        "severity": "high",
                        "count": int(len(syscall_data)),
                        "avg_latency": float(syscall_data["latency_us"].mean()),
                        "recommendation": f"Consider batching or caching {syscall} calls. "
                        f"Average latency: {syscall_data['latency_us'].mean():.2f}μs",
                    }
                )

        # Redundant calls
        redundant = results[results["is_redundant"] == True]
        if len(redundant) > 0:
            for syscall in redundant["syscall_name"].unique()[:5]:
                syscall_data = redundant[redundant["syscall_name"] == syscall]
                recommendations.append(
                    {
                        "type": "redundancy",
                        "syscall": syscall,
                        "severity": "medium",
                        "count": int(len(syscall_data)),
                        "recommendation": f"Eliminate redundant {syscall} calls. "
                        f"Detected {len(syscall_data)} potentially redundant calls.",
                    }
                )

        # Anomalous patterns
        anomalies = results[results["is_anomaly"] == True]
        if len(anomalies) > 0:
            recommendations.append(
                {
                    "type": "anomaly",
                    "severity": "high",
                    "count": int(len(anomalies)),
                    "recommendation": f"Investigate {len(anomalies)} anomalous syscall patterns. "
                    f"These may indicate performance issues or bugs.",
                }
            )

        return recommendations, analysis

    def save_models(self, path="models"):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")

        os.makedirs(path, exist_ok=True)

        joblib.dump(self.latency_model, f"{path}/latency_model.pkl")
        joblib.dump(self.redundancy_model, f"{path}/redundancy_model.pkl")
        joblib.dump(self.anomaly_detector, f"{path}/anomaly_detector.pkl")
        joblib.dump(self.pattern_clusterer, f"{path}/pattern_clusterer.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")

        with open(f"{path}/config.json", "w") as f:
            json.dump(
                {
                    "feature_columns": self.feature_columns,
                    "training_metrics": self.training_metrics,
                },
                f,
                indent=2,
            )

        print(f"Models saved to {path}/")

    def load_models(self, path="models"):
        """Load trained models from disk"""
        self.latency_model = joblib.load(f"{path}/latency_model.pkl")
        self.redundancy_model = joblib.load(f"{path}/redundancy_model.pkl")
        self.anomaly_detector = joblib.load(f"{path}/anomaly_detector.pkl")
        self.pattern_clusterer = joblib.load(f"{path}/pattern_clusterer.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")

        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
            self.feature_columns = config["feature_columns"]
            self.training_metrics = config["training_metrics"]

        self.is_trained = True
        print(f"Models loaded from {path}/")


def main():
    """Example usage"""
    # Import data generator
    import sys

    sys.path.append("..")
    from data.syscall_data_generator import SyscallDataGenerator

    # Generate training data
    generator = SyscallDataGenerator()
    training_data = generator.generate_training_dataset(samples_per_type=5000)

    # Train analyzer
    analyzer = SyscallAnalyzer()
    metrics = analyzer.train(training_data)

    print("\n=== Training Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Save models
    analyzer.save_models("trained_models")

    # Generate test data and make predictions
    test_data = generator.generate_syscall_trace(1000, "mixed")
    recommendations, analysis = analyzer.get_optimization_recommendations(test_data)

    print("\n=== Optimization Recommendations ===")
    for rec in recommendations:
        print(f"\n[{rec['severity'].upper()}] {rec['type']}")
        print(f"  {rec['recommendation']}")


if __name__ == "__main__":
    main()
