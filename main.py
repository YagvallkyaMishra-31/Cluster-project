"""Run the complete PySpark insurance claim risk workflow and build deliverables."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LEAKAGE_COLUMNS = {
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "incident_severity",
    "fraud_reported",
    "risk_score",
    "risk_level",
    "policy_number",
    "insured_zip",
    "incident_location",
}


class _Tee:
    """Write console output both to the terminal and the run log."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _configure_windows_hadoop() -> None:
    """Configure local Hadoop paths when available and warn when winutils is missing."""
    hadoop_root = Path(r"C:\hadoop")
    if not os.environ.get("HADOOP_HOME") and hadoop_root.is_dir():
        os.environ["HADOOP_HOME"] = str(hadoop_root)
        bin_path = str(hadoop_root / "bin")
        current_path = os.environ.get("PATH", "")
        if bin_path.lower() not in {part.lower() for part in current_path.split(os.pathsep)}:
            os.environ["PATH"] = bin_path + os.pathsep + current_path

    configured_home = os.environ.get("HADOOP_HOME")
    winutils = Path(configured_home) / "bin" / "winutils.exe" if configured_home else None
    available = bool(winutils and winutils.is_file()) or shutil.which("winutils.exe") is not None
    if not available:
        print(
            "WARNING: winutils.exe not found; Spark model persistence may fail on Windows, "
            "but CSV/report/dashboard exports will continue."
        )


def _overfit_comment(gap: float) -> str:
    if gap > 0.10:
        return f"The train score is notably higher (gap {gap:.4f}), suggesting possible overfitting."
    if gap > 0:
        return f"The train score is higher by {gap:.4f}; this is a modest train/test gap."
    if gap < 0:
        return f"The test score is higher by {abs(gap):.4f}; no positive train-over-test gap is observed."
    return "Training and test weighted F1 are equal to the displayed precision."


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    _configure_windows_hadoop()
    root = Path(__file__).resolve().parent
    log_path = root / "outputs" / "run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)  # type: ignore[assignment]
    spark = None

    try:
        # Keep Spark's driver and Python workers on the same interpreter.
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        from pyspark.sql import SparkSession

        from src.data_loader import clean_and_label, data_dictionary, inspect_dataset, load_claims, missing_summary
        from src.deliverables import create_project_deliverables
        from src.pipeline import train_evaluate
        from src.visualize import create_visualizations
        import pandas as pd

        data_path = root / "Data" / "insurance_claims.csv"
        outputs_dir = root / "outputs"
        data_processed_dir = root / "data" / "processed"
        dashboard_dir = root / "dashboard"
        powerbi_dir = dashboard_dir / "powerbi_data"
        for directory in (outputs_dir, data_processed_dir, dashboard_dir, powerbi_dir, root / "report", root / "notebooks"):
            directory.mkdir(parents=True, exist_ok=True)

        if not data_path.is_file():
            raise FileNotFoundError(f"Required authentic dataset is missing: {data_path}")

        spark = (
            SparkSession.builder.appName("InsuranceClaimRiskAnalysis")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.default.parallelism", "4")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        print("\n========== TASK 1: LOAD AND INSPECT ==========")
        print(f"Authentic source dataset: {data_path}")
        source = load_claims(spark, data_path)
        inspect_dataset(source)

        print("\n========== TASK 2: CLEAN, NORMALIZE, AND LABEL ==========")
        cleaned, cleaning_metadata = clean_and_label(source)
        cleaned_pdf = cleaned.toPandas()
        cleaned_path = data_processed_dir / "cleaned_claims.csv"
        cleaned_pdf.to_csv(cleaned_path, index=False)
        dictionary_pdf = pd.DataFrame(data_dictionary(cleaned, missing_summary(cleaned)))
        dictionary_pdf.to_csv(outputs_dir / "data_dictionary.csv", index=False)

        results = train_evaluate(
            spark,
            cleaned,
            cleaning_metadata,
            outputs_dir,
        )
        accuracy = float(results["metrics"]["accuracy"])
        if accuracy > 0.90:
            audit_text = (
                f"ALERT: held-out accuracy={accuracy:.4f} exceeds 0.90. "
                "The model-input leakage exclusion assertion passed for the configured fields, "
                "but this result must be investigated before use. Review transformed feature "
                "metadata, the label construction, partitioning, duplicates, and any unlisted "
                "target proxies. Configured excluded fields: "
                + ", ".join(sorted(LEAKAGE_COLUMNS))
                + "\n"
            )
        else:
            audit_text = (
                f"Held-out accuracy={accuracy:.4f}, not above the 0.90 investigation threshold. "
                "The configured leakage exclusion assertion passed. This does not constitute "
                "proof against unknown proxies or other leakage.\n"
            )
        (outputs_dir / "leakage_audit.txt").write_text(audit_text, encoding="utf-8")
        summaries = create_visualizations(
            cleaned,
            results,
            outputs_dir / "figures",
            dashboard_dir,
        )

        predictions_path = data_processed_dir / "cleaned_claims_with_predictions.csv"
        if not predictions_path.is_file():
            raise FileNotFoundError(f"Prediction export was not generated: {predictions_path}")
        predictions_pdf = pd.read_csv(predictions_path)
        predictions_pdf.to_csv(powerbi_dir / "cleaned_claims_with_predictions.csv", index=False)
        cleaned_pdf.to_csv(powerbi_dir / "cleaned_claims.csv", index=False)

        metrics_frame = pd.read_csv(outputs_dir / "metrics.csv")
        per_class = pd.read_csv(outputs_dir / "per_class_metrics.csv")
        baseline_per_class = pd.read_csv(outputs_dir / "baseline_per_class_metrics.csv")
        confusion_matrix = pd.read_csv(
            outputs_dir / "confusion_matrix.csv", index_col=0
        )
        feature_importance = pd.read_csv(outputs_dir / "feature_importance.csv")
        cv_results = pd.read_csv(outputs_dir / "cross_validation_results.csv")
        for filename, frame in (
            ("metrics.csv", metrics_frame),
            ("per_class_metrics.csv", per_class),
            ("baseline_per_class_metrics.csv", baseline_per_class),
            ("confusion_matrix.csv", confusion_matrix.reset_index()),
            ("feature_importance.csv", feature_importance),
            ("cross_validation_results.csv", cv_results),
        ):
            frame.to_csv(powerbi_dir / filename, index=False)
        dictionary_pdf.to_csv(powerbi_dir / "data_dictionary.csv", index=False)

        risk_counts = {
            row["risk_level"]: int(row["count"])
            for row in cleaned.groupBy("risk_level").count().collect()
        }
        source_rows = int(cleaning_metadata["source_rows"])
        cleaned_rows = int(cleaning_metadata["cleaned_rows"])
        run_data: Dict[str, Any] = {
            "source_rows": source_rows,
            "cleaned_rows": cleaned_rows,
            "risk_counts": risk_counts,
            "risk_score_quantiles": cleaning_metadata["risk_score_quantiles"],
            "claim_amount_iqr": cleaning_metadata["claim_amount_iqr"],
            "numeric_model_features": cleaning_metadata["numeric_model_features"],
            "categorical_model_features": cleaning_metadata["categorical_model_features"],
            "feature_columns": results["feature_columns"],
            "best_params": results["best_params"],
            "metrics": results["metrics"],
            "train_metrics": results["train_metrics"],
            "baseline_metrics": results["baseline_metrics"],
            "majority_baseline": results["majority_baseline"],
            "use_weights": results["use_weights"],
            "per_class_metrics": per_class.to_dict(orient="records"),
            "baseline_per_class_metrics": baseline_per_class.to_dict(orient="records"),
            "overfit_gap": results["overfit_gap"],
            "overfit_comment": _overfit_comment(results["overfit_gap"]),
            "weighted_precision_note": results["weighted_precision_note"],
            "model_save_error": results["model_save_error"],
            "mappings": {
                "claim_type": "incident_type",
                "customer_region": "policy_state",
                "policy_type": "policy_csl (coverage-limit tier proxy)",
            },
            "risk_label_rule": (
                "risk_score = 0.5 * normalized total_claim_amount + "
                "0.3 * normalized incident severity rank + 0.2 * fraud indicator; "
                "Low <= p40, Medium <= p75, otherwise High."
            ),
            "leakage_columns": sorted(LEAKAGE_COLUMNS),
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": "Data/insurance_claims.csv",
            "model_saved": results["model_save_error"] is None,
        }
        _write_json(outputs_dir / "run_metadata.json", run_data)

        deliverables_run: Dict[str, Any] = {
            **run_data,
            "metrics_frame": metrics_frame,
            "per_class": per_class,
            "baseline_per_class": baseline_per_class,
            "confusion_matrix": confusion_matrix,
            "feature_importance": feature_importance,
            "cv_results": cv_results,
            "summaries": summaries,
            "data_dictionary": dictionary_pdf,
            "use_weights": results["use_weights"],
            "model_save_error": results["model_save_error"],
            "overfit_gap": results["overfit_gap"],
            "overfit_comment": _overfit_comment(results["overfit_gap"]),
            "weighted_precision_note": results["weighted_precision_note"],
        }
        create_project_deliverables(root, deliverables_run)
        print("\n========== RUN COMPLETE ==========")
        print(f"Cleaned records: {cleaned_rows:,}")
        print(f"Risk counts: {risk_counts}")
        print(f"Best CV parameters: {results['best_params']}")
        print(f"Tuned held-out metrics: {results['metrics']}")
        print(f"Model saved: {results['model_save_error'] is None}")
        print(f"Results, reports, and dashboard written under: {root}")
        return 0
    except Exception as exc:
        print(f"ERROR: workflow did not complete: {type(exc).__name__}: {exc}")
        raise
    finally:
        if spark is not None:
            spark.stop()
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
