"""Verify that the generated project deliverables are complete and internally consistent."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from pptx import Presentation
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"Low", "Medium", "High"}
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


def main() -> int:
    required = [
        "README.md",
        "main.py",
        "outputs/run_metadata.json",
        "outputs/run_log.txt",
        "outputs/metrics.csv",
        "outputs/per_class_metrics.csv",
        "outputs/confusion_matrix.csv",
        "outputs/feature_importance.csv",
        "outputs/cross_validation_results.csv",
        "outputs/classification_report.txt",
        "outputs/decision_tree.txt",
        "outputs/pipeline_diagram.png",
        "outputs/figures/confusion_matrix.png",
        "outputs/figures/feature_importance.png",
        "outputs/figures/risk_distribution.png",
        "outputs/figures/policy_type_risk_heatmap.png",
        "outputs/figures/customer_region_risk_heatmap.png",
        "outputs/figures/claim_type_risk_heatmap.png",
        "outputs/figures/risk_by_policy_type_counts.png",
        "outputs/figures/risk_by_policy_type_percent_stacked.png",
        "outputs/figures/risk_by_region_counts.png",
        "outputs/figures/risk_by_region_percent_stacked.png",
        "outputs/figures/risk_by_claim_type_counts.png",
        "outputs/figures/risk_by_claim_type_percent_stacked.png",
        "data/processed/cleaned_claims.csv",
        "data/processed/cleaned_claims_with_predictions.csv",
        "dashboard/Insurance_Claim_Risk_Dashboard.xlsx",
        "dashboard/dashboard_preview.png",
        "dashboard/POWERBI_GUIDE.md",
        "dashboard/powerbi_data/metrics.csv",
        "dashboard/powerbi_data/per_class_metrics.csv",
        "dashboard/powerbi_data/baseline_per_class_metrics.csv",
        "dashboard/powerbi_data/confusion_matrix.csv",
        "dashboard/powerbi_data/feature_importance.csv",
        "dashboard/powerbi_data/cross_validation_results.csv",
        "dashboard/powerbi_data/data_dictionary.csv",
        "dashboard/powerbi_data/risk_by_policy_type.csv",
        "dashboard/powerbi_data/risk_by_region.csv",
        "dashboard/powerbi_data/risk_by_claim_type.csv",
        "dashboard/powerbi_data/risk_distribution.csv",
        "dashboard/powerbi_data/claim_amount_distribution.csv",
        "dashboard/powerbi_data/cleaned_claims_with_predictions.csv",
        "notebooks/claim_risk_analysis.ipynb",
        "report/Insurance_Claim_Risk_Report.docx",
        "report/presentation.pptx",
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    if missing:
        raise AssertionError(f"Required deliverables are missing: {missing}")

    run = json.loads((ROOT / "outputs" / "run_metadata.json").read_text(encoding="utf-8"))
    cleaned = pd.read_csv(ROOT / "data" / "processed" / "cleaned_claims.csv")
    predictions = pd.read_csv(
        ROOT / "data" / "processed" / "cleaned_claims_with_predictions.csv"
    )
    if int(cleaned.isna().sum().sum()) != 0:
        raise AssertionError("Cleaned data contains null values.")
    if int(predictions.isna().sum().sum()) != 0:
        raise AssertionError("Cleaned data with predictions contains null values.")
    for column in ("risk_level", "predicted_risk_level"):
        observed = set(predictions[column].dropna().astype(str))
        if not observed <= LABELS:
            raise AssertionError(f"Unexpected labels in {column}: {observed - LABELS}")
    if set(predictions["risk_level"].dropna().astype(str)) != LABELS:
        raise AssertionError("Cleaned target does not contain all three risk classes.")

    used_features = set(run["feature_columns"])
    leaked = used_features & LEAKAGE_COLUMNS
    if leaked:
        raise AssertionError(f"Leakage columns are present in model features: {sorted(leaked)}")
    if len(predictions) != run["cleaned_rows"]:
        raise AssertionError(
            f"Prediction count {len(predictions)} != cleaned count {run['cleaned_rows']}."
        )

    notebook = nbformat.read(ROOT / "notebooks" / "claim_risk_analysis.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells or any(cell.execution_count is None for cell in code_cells):
        raise AssertionError("Notebook code cells were not all executed.")
    for cell in code_cells:
        if any(output.output_type == "error" for output in cell.outputs):
            raise AssertionError("Notebook contains a failed code cell.")

    workbook = load_workbook(ROOT / "dashboard" / "Insurance_Claim_Risk_Dashboard.xlsx", read_only=False)
    expected_sheets = {"Dashboard", "Dataset Description", "Summary tables", "Cleaned data"}
    if not expected_sheets <= set(workbook.sheetnames):
        raise AssertionError(f"Workbook sheets missing: {expected_sheets - set(workbook.sheetnames)}")
    if len(workbook["Dashboard"]._charts) < 1:
        raise AssertionError("Dashboard workbook has no native Excel charts.")

    slides = Presentation(ROOT / "report" / "presentation.pptx")
    if not 10 <= len(slides.slides) <= 12:
        raise AssertionError(f"Expected 10-12 presentation slides, found {len(slides.slides)}.")
    if any(not slide.has_notes_slide for slide in slides.slides):
        raise AssertionError("Presentation speaker notes are missing.")

    metrics = pd.read_csv(ROOT / "outputs" / "metrics.csv")
    if not {"majority_class_baseline", "untuned_tree", "tuned_tree"} <= set(metrics["model"]):
        raise AssertionError("Model comparison table is missing required models.")
    if set(pd.read_csv(ROOT / "outputs" / "per_class_metrics.csv")["class"]) != LABELS:
        raise AssertionError("Per-class metrics do not contain Low, Medium, and High.")
    if len(pd.read_csv(ROOT / "outputs" / "cross_validation_results.csv")) != 16:
        raise AssertionError("Cross-validation results do not contain the expected parameter grid.")
    print(
        "Smoke test passed: required outputs, zero-null cleaned data, allowed labels, "
        "leakage exclusions, executed notebook, workbook charts/sheets, and presentation notes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
