"""Read, inspect, and prepare the authentic insurance claims CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

SEVERITY_ORDER = ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
TARGET_DERIVED_COLUMNS = {
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "incident_severity",
    "fraud_reported",
    "risk_score",
    "risk_level",
}
IDENTIFIER_COLUMNS = {"policy_number", "insured_zip", "incident_location"}


def load_claims(spark: SparkSession, csv_path: Path) -> DataFrame:
    """Read the raw CSV as strings so question marks can be handled consistently."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required authentic dataset is missing: {csv_path}")
    frame = (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "FAILFAST")
        .csv(str(csv_path))
    )
    empty_columns = [
        name
        for name in frame.columns
        if frame.where(F.col(name).isNotNull() & (F.trim(F.col(name)) != "")).limit(1).count() == 0
    ]
    if empty_columns:
        frame = frame.drop(*empty_columns)
        print(f"Dropped empty trailing columns: {empty_columns}")
    return frame


def missing_summary(frame: DataFrame) -> DataFrame:
    """Return missing counts and percentages for every column."""
    total = frame.count()
    rows = []
    for name in frame.columns:
        missing = frame.where(
            F.col(name).isNull()
            | (F.trim(F.col(name).cast("string")) == "")
            | (F.trim(F.col(name).cast("string")) == "?")
        ).count()
        rows.append((name, missing, round((missing / total * 100) if total else 0.0, 4)))
    return frame.sparkSession.createDataFrame(rows, ["column", "missing_count", "missing_percent"])


def inspect_dataset(frame: DataFrame) -> None:
    """Print the required initial schema, size, examples, and summary."""
    print("\n========== TASK 1: LOAD AND INSPECT ==========")
    frame.printSchema()
    print(f"Row count: {frame.count():,}")
    print("Sample rows:")
    frame.show(5, truncate=False)
    print("Describe:")
    frame.describe().show(truncate=False)
    print("Question-mark value counts:")
    for name in frame.columns:
        if frame.where(F.trim(F.col(name)) == "?").limit(1).count():
            print(f"  {name}")
            frame.groupBy(name).count().orderBy(F.desc("count")).show(truncate=False)


def _cast_numeric_columns(frame: DataFrame) -> Tuple[DataFrame, List[str]]:
    """Cast columns that are numeric in the inspected source."""
    numeric_names = []
    for name in frame.columns:
        if name in {"policy_bind_date", "incident_date"}:
            continue
        if frame.where(
            F.col(name).isNotNull()
            & (F.trim(F.col(name)) != "")
            & F.col(name).cast("double").isNull()
        ).limit(1).count() == 0:
            if frame.where(F.col(name).isNotNull() & (F.trim(F.col(name)) != "")).limit(1).count():
                frame = frame.withColumn(name, F.col(name).cast("double"))
                numeric_names.append(name)
    return frame, numeric_names


def clean_and_label(frame: DataFrame) -> Tuple[DataFrame, Dict[str, object]]:
    """Clean missing markers, cap claim outliers, and derive the requested risk label."""
    print("\n========== TASK 2: CLEAN, NORMALIZE, AND LABEL ==========")
    print("Missing values before cleaning (including literal question marks):")
    missing_summary(frame).show(100, truncate=False)
    cleaned = frame.select(
        *[
            F.when(F.trim(F.col(name).cast("string")) == "?", None)
            .when(F.trim(F.col(name).cast("string")) == "", None)
            .otherwise(F.col(name))
            .alias(name)
            for name in frame.columns
            if name != "_c39"
        ]
    )
    cleaned = cleaned.dropDuplicates()
    cleaned, numeric_columns = _cast_numeric_columns(cleaned)
    if "total_claim_amount" not in cleaned.columns:
        raise ValueError("The required total_claim_amount column is missing.")

    q1, q3 = cleaned.approxQuantile("total_claim_amount", [0.25, 0.75], 0.001)
    iqr = q3 - q1
    lower_bound, upper_bound = max(0.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    cleaned = cleaned.withColumn(
        "total_claim_amount",
        F.when(F.col("total_claim_amount") < lower_bound, lower_bound)
        .when(F.col("total_claim_amount") > upper_bound, upper_bound)
        .otherwise(F.col("total_claim_amount")),
    )
    print(
        f"Claim amount IQR cap: Q1={q1:.2f}, Q3={q3:.2f}, "
        f"lower={lower_bound:.2f}, upper={upper_bound:.2f}."
    )

    for name in numeric_columns:
        if name not in TARGET_DERIVED_COLUMNS:
            median_values = cleaned.approxQuantile(name, [0.5], 0.001)
            fill_value = median_values[0] if median_values else 0.0
            cleaned = cleaned.withColumn(name, F.coalesce(F.col(name), F.lit(fill_value)))
    categorical = [
        name
        for name in cleaned.columns
        if name not in numeric_columns
    ]
    for name in categorical:
        cleaned = cleaned.withColumn(name, F.coalesce(F.col(name), F.lit("Unknown")))

    severity_mapping = F.create_map(
        *[
            value
            for index, severity in enumerate(SEVERITY_ORDER, start=1)
            for value in (F.lit(severity), F.lit(float(index)))
        ]
    )
    total_min, total_max = cleaned.agg(
        F.min("total_claim_amount"), F.max("total_claim_amount")
    ).first()
    if total_max == total_min:
        claim_component = F.lit(0.0)
    else:
        claim_component = (F.col("total_claim_amount") - F.lit(total_min)) / F.lit(
            total_max - total_min
        )
    cleaned = cleaned.withColumn(
        "_severity_rank", F.coalesce(severity_mapping[F.col("incident_severity")], F.lit(2.5))
    )
    rank_window = Window.orderBy("_severity_rank")
    cleaned = cleaned.withColumn(
        "_severity_rank_normalized",
        F.when(F.count(F.lit(1)).over(rank_window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)) <= 1, 0.0)
        .otherwise(F.percent_rank().over(rank_window)),
    )
    cleaned = cleaned.withColumn(
        "risk_score",
        0.5 * claim_component
        + 0.3 * F.col("_severity_rank_normalized")
        + 0.2
        * F.when(F.upper(F.col("fraud_reported").cast("string")) == "Y", 1.0).otherwise(0.0),
    )
    low_cut, high_cut = cleaned.approxQuantile("risk_score", [0.4, 0.75], 0.001)
    cleaned = cleaned.withColumn(
        "risk_level",
        F.when(F.col("risk_score") <= low_cut, "Low")
        .when(F.col("risk_score") <= high_cut, "Medium")
        .otherwise("High"),
    ).drop("_severity_rank", "_severity_rank_normalized")

    for source_name, mapped_name in (
        ("incident_type", "claim_type"),
        ("policy_state", "customer_region"),
        ("policy_csl", "policy_type"),
    ):
        if source_name not in cleaned.columns:
            raise ValueError(f"Required source column for {mapped_name} mapping is missing: {source_name}")
        cleaned = cleaned.withColumn(mapped_name, F.col(source_name))
    for source_name in ("policy_bind_date", "incident_date"):
        if source_name in cleaned.columns:
            date_column = F.to_date(F.col(source_name), "yyyy-MM-dd")
            cleaned = cleaned.withColumn(f"{source_name}_month", F.month(date_column).cast("double"))
            cleaned = cleaned.withColumn(f"{source_name}_year", F.year(date_column).cast("double"))

    # Model input categoricals use "Unknown"; model numeric imputation is fitted in the training pipeline.
    mapped_source_columns = {"incident_type", "policy_state", "policy_csl"}
    date_source_columns = {"policy_bind_date", "incident_date"}
    numeric_model_features = [
        name for name in numeric_columns
        if name not in TARGET_DERIVED_COLUMNS | IDENTIFIER_COLUMNS | mapped_source_columns | date_source_columns
    ]
    numeric_model_features.extend(
        name for name in (
            "policy_bind_date_month",
            "policy_bind_date_year",
            "incident_date_month",
            "incident_date_year",
        )
        if name in cleaned.columns
    )
    categorical_model_features = [
        name for name in cleaned.columns
        if name not in numeric_columns
        and name not in TARGET_DERIVED_COLUMNS | IDENTIFIER_COLUMNS | mapped_source_columns | date_source_columns
        and name not in numeric_model_features
    ]
    for name in categorical_model_features:
        cleaned = cleaned.withColumn(name, F.coalesce(F.col(name), F.lit("Unknown")))
    # Non-model columns are also made complete for EDA and the exported cleaned data.
    for name in cleaned.columns:
        if isinstance(cleaned.schema[name].dataType, NumericType):
            if name != "risk_score":
                medians = cleaned.approxQuantile(name, [0.5], 0.001)
                cleaned = cleaned.withColumn(
                    name, F.coalesce(F.col(name), F.lit(medians[0] if medians else 0.0))
                )
        elif name not in TARGET_DERIVED_COLUMNS and name not in IDENTIFIER_COLUMNS:
            cleaned = cleaned.withColumn(name, F.coalesce(F.col(name), F.lit("Unknown")))

    print("Missing values after cleaning:")
    missing_summary(cleaned).show(100, truncate=False)
    print("Risk distribution:")
    cleaned.groupBy("risk_level").count().orderBy("risk_level").show()
    print("Claim amount before/after IQR capping (cleaned distribution):")
    cleaned.select("total_claim_amount").describe().show()
    metadata = {
        "source_rows": frame.count(),
        "cleaned_rows": cleaned.count(),
        "columns": cleaned.columns,
        "numeric_columns": numeric_columns,
        "numeric_model_features": numeric_model_features,
        "categorical_model_features": categorical_model_features,
        "risk_score_quantiles": {"p40": low_cut, "p75": high_cut},
        "claim_amount_iqr": {
            "q1": q1,
            "q3": q3,
            "lower_cap": lower_bound,
            "upper_cap": upper_bound,
        },
        "total_claim_min": total_min,
        "total_claim_max": total_max,
        "duplicate_rows_removed": frame.count() - cleaned.count(),
    }
    return cleaned, metadata


def data_dictionary(frame: DataFrame, missing: DataFrame) -> List[Dict[str, object]]:
    """Create a small driver-side data dictionary for dashboard and report output."""
    types = dict(frame.dtypes)
    missing_by_name = {
        row["column"]: row["missing_percent"] for row in missing.collect()
    }
    descriptions = {
        "incident_type": "Claim type (source incident_type).",
        "policy_state": "Customer region proxy (source policy_state).",
        "policy_csl": "Coverage-limit tier proxy for policy type; the source has no policy-type field.",
        "claim_type": "Mapped from incident_type.",
        "customer_region": "Mapped from policy_state.",
        "policy_type": "Mapped from policy_csl (coverage-limit tier proxy).",
        "risk_level": "Derived target: Low, Medium, or High.",
        "total_claim_amount": "Total claim amount; used for the derived target, EDA, and display only.",
        "policy_number": "Policy identifier; excluded from model features.",
        "insured_zip": "Insured postal code; excluded from model features.",
        "incident_location": "Incident address; excluded from model features.",
    }
    return [
        {
            "name": name,
            "type": types.get(name, "unknown"),
            "description": descriptions.get(name, name.replace("_", " ").capitalize() + "."),
            "missing_percent": missing_by_name.get(name, 0.0),
        }
        for name in frame.columns
    ]
