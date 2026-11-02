# Power BI guide

## Import

1. Open Power BI Desktop and choose **Get data → Text/CSV**.
2. Import the CSV files in `dashboard/powerbi_data/`. Start with
   `risk_distribution.csv`, `risk_by_policy_type.csv`, `risk_by_region.csv`, and
   `risk_by_claim_type.csv`. `cleaned_claims_with_predictions.csv`,
   `metrics.csv`, `per_class_metrics.csv`, and `confusion_matrix.csv` provide additional
   model and record-level views.
3. Keep `risk_level` and `predicted_risk_level` as categorical/text fields. Treat `count`,
   `accuracy`, `precision`, `recall`, `f1`, and claim amount fields as numeric measures.

## Suggested visuals

- KPI cards: total cleaned rows, tuned test accuracy, tuned weighted F1, and high-risk count.
- Clustered/stacked column chart: `risk_by_policy_type.csv`, axis `policy_type`, legend
  risk columns Low/Medium/High.
- Region and claim-type charts: use the corresponding `risk_by_region.csv` and
  `risk_by_claim_type.csv` with the same risk series.
- Overall risk distribution: `risk_distribution.csv`, category `risk_level`, value `count`.
- Confusion matrix: `confusion_matrix.csv`, actual risk as row and predicted risk as column.

`policy_type` is a proxy mapped from source `policy_csl`, not a native policy category.
The risk label itself is a rule-derived target; it should not be presented as independently
verified fraud or underwriting truth.
