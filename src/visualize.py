"""Create Spark-aggregated risk summaries and publication-quality charts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2E8B57", "Medium": "#F39C12", "High": "#C0392B"}


def _risk_table(frame: DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate a dimension-by-risk table in Spark before collecting it."""
    result = (
        frame.groupBy(dimension)
        .pivot("risk_level", RISK_ORDER)
        .count()
        .fillna(0)
        .orderBy(dimension)
    )
    return result.toPandas()


def _plot_grouped(data: pd.DataFrame, dimension: str, path: Path, percent: bool = False) -> None:
    labels = data[dimension].astype(str)
    values = data.reindex(columns=RISK_ORDER, fill_value=0).copy()
    if percent:
        totals = values.sum(axis=1).replace(0, 1)
        values = values.div(totals, axis=0) * 100
    axis = values.set_index(labels).plot(
        kind="bar", color=[RISK_COLORS[label] for label in RISK_ORDER], figsize=(10, 5)
    )
    axis.set_title(f"{'Risk share' if percent else 'Claim counts'} by {dimension.replace('_', ' ').title()}")
    axis.set_xlabel(dimension.replace("_", " ").title())
    axis.set_ylabel("Percent of claims" if percent else "Claim count")
    if percent:
        axis.set_ylim(0, 100)
    axis.legend(title="Risk level")
    axis.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_stacked(data: pd.DataFrame, dimension: str, path: Path) -> None:
    labels = data[dimension].astype(str)
    values = data.reindex(columns=RISK_ORDER, fill_value=0).copy()
    totals = values.sum(axis=1).replace(0, 1)
    values = values.div(totals, axis=0) * 100
    axis = values.set_index(labels).plot(
        kind="bar",
        stacked=True,
        color=[RISK_COLORS[label] for label in RISK_ORDER],
        figsize=(10, 5),
    )
    axis.set_title(f"Risk composition by {dimension.replace('_', ' ').title()}")
    axis.set_xlabel(dimension.replace("_", " ").title())
    axis.set_ylabel("Percent of claims")
    axis.set_ylim(0, 100)
    axis.legend(title="Risk level")
    axis.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def create_visualizations(
    cleaned: DataFrame,
    results: Dict[str, object],
    output_dir: Path,
    dashboard_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Aggregate in Spark, save 300-dpi plots, and create dashboard preview."""
    print("\n========== TASK 6: VISUALIZE RISK DISTRIBUTIONS ==========")
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    powerbi_dir = dashboard_dir / "powerbi_data"
    powerbi_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for column, key in (
        ("policy_type", "risk_by_policy_type"),
        ("customer_region", "risk_by_region"),
        ("claim_type", "risk_by_claim_type"),
    ):
        table = _risk_table(cleaned, column)
        summaries[key] = table
        table.to_csv(powerbi_dir / f"{key}.csv", index=False)
        _plot_grouped(table, column, output_dir / f"{key}_counts.png")
        _plot_stacked(table, column, output_dir / f"{key}_percent_stacked.png")

    distribution = (
        cleaned.groupBy("risk_level")
        .count()
        .orderBy(F.when(F.col("risk_level") == "Low", 1)
                 .when(F.col("risk_level") == "Medium", 2).otherwise(3))
        .toPandas()
    )
    summaries["risk_distribution"] = distribution
    distribution.to_csv(powerbi_dir / "risk_distribution.csv", index=False)
    distribution.set_index("risk_level").reindex(RISK_ORDER)["count"].plot(
        kind="bar", color=[RISK_COLORS[level] for level in RISK_ORDER], figsize=(8, 5)
    )
    plt.title("Overall risk class distribution")
    plt.xlabel("Risk level")
    plt.ylabel("Claim count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "risk_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    amount_bins = (
        cleaned.withColumn(
            "claim_amount_bin",
            F.when(F.col("total_claim_amount") < 10000, "< $10k")
            .when(F.col("total_claim_amount") < 20000, "$10k-$20k")
            .when(F.col("total_claim_amount") < 30000, "$20k-$30k")
            .otherwise("$30k+"),
        )
        .groupBy("claim_amount_bin")
        .count()
        .toPandas()
    )
    summaries["claim_amount_distribution"] = amount_bins
    amount_bins.to_csv(powerbi_dir / "claim_amount_distribution.csv", index=False)

    for column, table in (
        ("policy_type", summaries["risk_by_policy_type"]),
        ("customer_region", summaries["risk_by_region"]),
        ("claim_type", summaries["risk_by_claim_type"]),
    ):
        heat = table.set_index(column).reindex(columns=RISK_ORDER, fill_value=0)
        plt.figure(figsize=(8, max(3, len(heat) * 0.6)))
        sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd")
        plt.title(f"Claim count heatmap: {column.replace('_', ' ').title()} vs risk")
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_risk_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    confusion = results["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Test-set confusion matrix")
    plt.ylabel("Actual risk")
    plt.xlabel("Predicted risk")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance = results["feature_importance"].head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(importance["feature"], importance["importance"], color="#4472C4")
    plt.title("Top 15 decision-tree feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, key, dimension in (
        (axes[0, 0], "risk_by_policy_type", "policy_type"),
        (axes[0, 1], "risk_by_region", "customer_region"),
        (axes[1, 0], "risk_by_claim_type", "claim_type"),
    ):
        table = summaries[key]
        table.set_index(dimension).reindex(columns=RISK_ORDER, fill_value=0).plot(
            kind="bar", stacked=True, color=[RISK_COLORS[label] for label in RISK_ORDER], ax=axis
        )
        axis.set_title(f"Risk counts by {dimension.replace('_', ' ')}")
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelrotation=25)
        axis.legend(title="Risk", fontsize=8)
    axes[1, 1].bar(
        distribution["risk_level"],
        distribution["count"],
        color=[RISK_COLORS.get(level, "#777777") for level in distribution["risk_level"]],
    )
    axes[1, 1].set_title("Overall risk distribution")
    axes[1, 1].set_xlabel("Risk level")
    axes[1, 1].set_ylabel("Claims")
    fig.suptitle("Insurance Claim Risk Analysis — dashboard preview", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dashboard_dir / "dashboard_preview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summaries
