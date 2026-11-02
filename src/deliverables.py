"""Create project reports and dashboards from the actual pipeline results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from nbclient import NotebookClient
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches as PptInches
from pptx.util import Pt as PptPt

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2E8B57", "Medium": "#F39C12", "High": "#C0392B"}
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


def _format_metric(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return str(value)


def _table(document: Document, frame: pd.DataFrame) -> None:
    table = document.add_table(rows=1, cols=len(frame.columns))
    table.style = "Light Shading Accent 1"
    for cell, name in zip(table.rows[0].cells, frame.columns):
        cell.text = str(name)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = _format_metric(value)


def create_pipeline_diagram(path: Path) -> None:
    """Render the project pipeline diagram used in the report and presentation."""
    labels = [
        "Authentic CSV\n(1,000 claims)",
        "Inspect + clean\nmissing / duplicates",
        "Derive risk score\nand risk label",
        "Leakage-safe\nfeature pipeline",
        "Decision tree\n+ 3-fold CV",
        "Evaluate + export\nreports / dashboard",
    ]
    fig, axis = plt.subplots(figsize=(14, 3.2))
    axis.set_xlim(0, len(labels))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, label in enumerate(labels):
        axis.text(
            index + 0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#17324D",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#EAF2F8", "edgecolor": "#4472C4"},
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(index + 0.96, 0.5),
                xytext=(index + 0.75, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#4472C4"},
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_readme(root: Path, run: Dict[str, Any]) -> None:
    metrics = run["metrics_frame"].set_index("model")
    tuned = metrics.loc["tuned_tree"]
    baseline = metrics.loc["untuned_tree"]
    majority = metrics.loc["majority_class_baseline"]
    class_report = run["per_class"].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    model_status = (
        "Saved and reloaded successfully."
        if not run["model_save_error"]
        else f"Not saved in this Windows run: `{run['model_save_error']}`. "
        "This is non-fatal; the metrics, CSVs, reports, and visualizations were still generated."
    )
    readme = f"""# Insurance Claim Risk Analysis

## Project status and actual run

The workflow ran against the authentic source file `Data/insurance_claims.csv` with
{run['source_rows']:,} source rows and {run['cleaned_rows']:,} cleaned rows. The model
features intentionally exclude the derived target ingredients and identifier-like columns.
The best cross-validation tree parameters were `{json.dumps(run['best_params'], sort_keys=True)}`.

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Majority-class baseline ({majority['majority_class']}) | {majority['accuracy']:.4f} | {majority['weighted_precision']:.4f} | {majority['weighted_recall']:.4f} | {majority['weighted_f1']:.4f} |
| Untuned tree | {baseline['accuracy']:.4f} | {baseline['weighted_precision']:.4f} | {baseline['weighted_recall']:.4f} | {baseline['weighted_f1']:.4f} |
| Tuned tree | {tuned['accuracy']:.4f} | {tuned['weighted_precision']:.4f} | {tuned['weighted_recall']:.4f} | {tuned['weighted_f1']:.4f} |

Tuned-tree weighted F1 was {run['train_metrics']['weighted_f1']:.4f} on training data and
{tuned['weighted_f1']:.4f} on the held-out test partition (train minus test:
{run['overfit_gap']:.4f}). {run['overfit_comment']}

### Per-class held-out metrics

{class_report}

### Precision versus recall

{run['weighted_precision_note']}

## Setup and run

1. Use Python 3.11 and install the pinned/project dependencies:
   `python -m pip install -r requirements.txt`
2. Ensure Java and Spark 3.5 are installed and `SPARK_HOME` points to the Spark
   distribution. The project reads the authentic CSV from `Data/insurance_claims.csv`.
3. On Windows, Hadoop's `winutils.exe` may be required for Spark local filesystem
   permissions. If `C:\\hadoop\\bin\\winutils.exe` exists, `main.py` configures
   `HADOOP_HOME=C:\\hadoop` when it is unset. If it is absent, the runner prints a warning
   and continues; model persistence may fail, but it is caught and does not prevent other
   deliverables. Small deliverables use pandas CSV/Excel writers, not Spark filesystem writes.
4. Run from the project root: `python main.py`.
5. Optional verification: `python tests/smoke_test.py`.

## Implemented pipeline (six tasks)

1. **Load and inspect:** Spark reads the authentic CSV as strings, checks the schema,
   examples, row count, descriptive summary, question-mark values, and missingness.
2. **Clean and label:** question marks/blanks become nulls; duplicate rows are removed;
   numeric types and missing values are handled; total claim amount is IQR-capped; the
   risk score and Low/Medium/High labels are derived; date-part and category mappings are added.
3. **Encode:** a leakage-controlled Spark ML pipeline imputes numeric values, indexes and
   one-hot encodes categories, assembles and scales features.
4. **Train and tune:** a decision-tree baseline is compared with three-fold cross-validation
   over the recorded parameter grid. Training-only class weights are applied when needed.
5. **Evaluate:** majority baseline, baseline tree, tuned tree, per-class metrics, confusion
   matrix, feature importances, and train/test F1 are exported.
6. **Visualize:** grouped counts, 100% stacked risk shares, policy/region/claim heatmaps,
   overall class distribution, and dashboard preview are generated at 300 dpi.

## Mappings and target definition

- `claim_type = incident_type`
- `customer_region = policy_state`
- `policy_type = policy_csl` (coverage-limit tier proxy; the source has no policy-type column)
- `risk_score = 0.5 * normalized total_claim_amount + 0.3 * normalized incident-severity rank
  + 0.2 * fraud indicator (1 for Y, otherwise 0)`.
- Risk labels use the run's empirical score quantiles: Low at or below p40, Medium above p40
  through p75, and High above p75. Actual cut points: p40={run['risk_score_quantiles']['p40']:.6f},
  p75={run['risk_score_quantiles']['p75']:.6f}.

## Leakage controls

The model input assertion rejects these fields: `{', '.join(sorted(LEAKAGE_COLUMNS))}`.
These contain the target, target ingredients, claim amounts, or identifiers. The derived
risk label is a heuristic target, not a ground-truth future outcome; model scores measure
agreement with that constructed rule, not proven real-world fraud or loss prediction.
If held-out test accuracy exceeds 0.90, treat it as a leakage signal and investigate the
feature list, transformations, and split before interpreting the result. Actual tuned
test accuracy in this run: {tuned['accuracy']:.4f}.

## Model persistence

{model_status}

## Generated files

- `outputs/`: metrics, per-class and baseline per-class reports, confusion matrix, feature
  importances, cross-validation results, classification report, decision tree text, and charts.
- `data/processed/cleaned_claims_with_predictions.csv`: cleaned records plus actual
  predictions and probability vectors.
- `dashboard/Insurance_Claim_Risk_Dashboard.xlsx`: dashboard, dataset description, summary
  tables, cleaned data with predictions, and native Excel charts.
- `dashboard/powerbi_data/`: CSVs suitable for Power BI; see `dashboard/POWERBI_GUIDE.md`.
- `dashboard/dashboard_preview.png`: summary visualization used in the workbook and reports.
- `notebooks/claim_risk_analysis.ipynb`: executed analysis notebook based on the saved run outputs.
- `report/Insurance_Claim_Risk_Report.docx` and `report/presentation.pptx`: report and
  presentation with actual run output, findings, limitations, references, and viva Q&A.

## Limitations

This is a 1,000-row dataset and an exploratory decision-tree workflow. The risk labels are
constructed from claim amount, incident severity, and a fraud indicator; therefore those
fields are excluded from model inputs, but the resulting target is still heuristic. The
random held-out split is not a temporal or external validation. `policy_type` is only a
proxy for coverage-limit tier. A saved model is not guaranteed on Windows without a
working Hadoop `winutils.exe` setup.

## References

- Provided source data: `Data/insurance_claims.csv`.
- Apache Spark 3.5 documentation: https://spark.apache.org/docs/3.5.7/
- PySpark ML guide: https://spark.apache.org/docs/3.5.7/ml-guide.html
- pandas documentation: https://pandas.pydata.org/docs/
- XlsxWriter documentation: https://xlsxwriter.readthedocs.io/
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def _write_powerbi_guide(root: Path) -> None:
    guide = """# Power BI guide

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
"""
    (root / "dashboard" / "POWERBI_GUIDE.md").write_text(guide, encoding="utf-8")


def _write_excel_dashboard(root: Path, run: Dict[str, Any]) -> None:
    path = root / "dashboard" / "Insurance_Claim_Risk_Dashboard.xlsx"
    dashboard_dir = path.parent
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    cleaned = pd.read_csv(root / "data" / "processed" / "cleaned_claims_with_predictions.csv")
    metrics = run["metrics_frame"]
    summary_tables = [
        ("Overall metrics comparison", metrics),
        ("Tuned per-class metrics", run["per_class"]),
        ("Baseline per-class metrics", run["baseline_per_class"]),
        ("Confusion matrix", run["confusion_matrix"].reset_index()),
        ("Cross-validation results", run["cv_results"]),
        ("Feature importances", run["feature_importance"]),
        ("Risk by policy type", run["summaries"]["risk_by_policy_type"]),
        ("Risk by region", run["summaries"]["risk_by_region"]),
        ("Risk by claim type", run["summaries"]["risk_by_claim_type"]),
        ("Overall risk distribution", run["summaries"]["risk_distribution"]),
    ]
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_format = workbook.add_format(
            {"bold": True, "font_size": 18, "font_color": "#17324D", "valign": "vcenter"}
        )
        card_title = workbook.add_format(
            {"bold": True, "font_color": "white", "bg_color": "#4472C4", "align": "center"}
        )
        card_value = workbook.add_format(
            {"bold": True, "font_size": 16, "font_color": "#17324D", "align": "center"}
        )
        section_format = workbook.add_format({"bold": True, "font_size": 12, "font_color": "#17324D"})
        dashboard = workbook.add_worksheet("Dashboard")
        writer.sheets["Dashboard"] = dashboard
        dashboard.merge_range("B2:I2", "Insurance Claim Risk Analysis", title_format)
        dashboard.set_row(1, 30)
        tuned = metrics.set_index("model").loc["tuned_tree"]
        cards = [
            ("Cleaned records", run["cleaned_rows"]),
            ("Tuned test accuracy", tuned["accuracy"]),
            ("Tuned weighted F1", tuned["weighted_f1"]),
            ("High-risk records", int(run["risk_counts"].get("High", 0))),
        ]
        for index, (label, value) in enumerate(cards):
            column = 1 + index * 2
            dashboard.merge_range(3, column, 3, column + 1, label, card_title)
            display_value = f"{value:.4f}" if isinstance(value, float) else f"{value:,}"
            dashboard.merge_range(4, column, 5, column + 1, display_value, card_value)
        dashboard.write("B8", "Native Excel charts and generated preview", section_format)

        dataset_info = pd.DataFrame(
            [
                {
                    "item": "Risk-label rule",
                    "detail": (
                        "0.5 * normalized total_claim_amount + 0.3 * normalized incident severity "
                        "+ 0.2 * fraud indicator; Low <= p40, Medium <= p75, else High."
                    ),
                },
                {"item": "claim_type mapping", "detail": "incident_type"},
                {"item": "customer_region mapping", "detail": "policy_state"},
                {"item": "policy_type mapping", "detail": "policy_csl coverage-limit proxy"},
            ]
        )
        description = run["data_dictionary"].copy()
        description = description.rename(
            columns={"name": "field", "type": "source_type", "description": "meaning"}
        )
        description.to_excel(writer, sheet_name="Dataset Description", index=False, startrow=8)
        dataset_info.to_excel(writer, sheet_name="Dataset Description", index=False)
        dataset_sheet = writer.sheets["Dataset Description"]
        dataset_sheet.freeze_panes(9, 0)
        dataset_sheet.set_column("A:A", 28)
        dataset_sheet.set_column("B:B", 55)
        dataset_sheet.set_column("C:D", 22)
        dataset_sheet.set_column("E:E", 80)

        summary_sheet = workbook.add_worksheet("Summary tables")
        writer.sheets["Summary tables"] = summary_sheet
        row_map: Dict[str, int] = {}
        current_row = 0
        for title, frame in summary_tables:
            summary_sheet.write(current_row, 0, title, section_format)
            data_row = current_row + 1
            frame.to_excel(
                writer,
                sheet_name="Summary tables",
                index=False,
                startrow=data_row,
            )
            row_map[title] = data_row
            current_row = data_row + len(frame) + 3
        summary_sheet.freeze_panes(1, 0)
        summary_sheet.set_column("A:A", 32)
        summary_sheet.set_column("B:Z", 20)
        clean_sheet = workbook.add_worksheet("Cleaned data")
        writer.sheets["Cleaned data"] = clean_sheet
        cleaned.to_excel(writer, sheet_name="Cleaned data", index=False)
        clean_sheet.freeze_panes(1, 0)
        clean_sheet.autofilter(0, 0, len(cleaned), max(len(cleaned.columns) - 1, 0))
        clean_sheet.set_column(0, max(len(cleaned.columns) - 1, 0), 16)

        risk_row = row_map["Overall risk distribution"]
        risk_frame = run["summaries"]["risk_distribution"].reset_index(drop=True)
        risk_headers = list(risk_frame.columns)
        risk_cols = {name: index for index, name in enumerate(risk_headers)}
        risk_chart = workbook.add_chart({"type": "column"})
        risk_chart.add_series(
            {
                "name": "Claim count",
                "categories": ["Summary tables", risk_row + 1, risk_cols["risk_level"], risk_row + len(risk_frame), risk_cols["risk_level"]],
                "values": ["Summary tables", risk_row + 1, risk_cols["count"], risk_row + len(risk_frame), risk_cols["count"]],
                "points": [
                    {"fill": {"color": RISK_COLORS[level]}}
                    for level in run["summaries"]["risk_distribution"]["risk_level"]
                ],
            }
        )
        risk_chart.set_title({"name": "Overall risk distribution"})
        risk_chart.set_y_axis({"name": "Claims"})
        risk_chart.set_legend({"none": True})
        dashboard.insert_chart("B10", risk_chart, {"x_scale": 1.15, "y_scale": 1.1})

        policy_frame = run["summaries"]["risk_by_policy_type"]
        policy_row = row_map["Risk by policy type"]
        policy_chart = workbook.add_chart({"type": "column", "subtype": "clustered"})
        for level in RISK_ORDER:
            if level not in policy_frame.columns:
                continue
            col = list(policy_frame.columns).index(level)
            policy_chart.add_series(
                {
                    "name": ["Summary tables", policy_row, col],
                    "categories": ["Summary tables", policy_row + 1, 0, policy_row + len(policy_frame), 0],
                    "values": ["Summary tables", policy_row + 1, col, policy_row + len(policy_frame), col],
                    "fill": {"color": RISK_COLORS[level]},
                }
            )
        policy_chart.set_title({"name": "Claims by policy type and risk"})
        policy_chart.set_y_axis({"name": "Claims"})
        dashboard.insert_chart("H10", policy_chart, {"x_scale": 1.15, "y_scale": 1.1})
        preview = dashboard_dir / "dashboard_preview.png"
        if preview.is_file():
            dashboard.insert_image("B28", str(preview), {"x_scale": 0.52, "y_scale": 0.52})
        dashboard.set_column("A:A", 3)
        dashboard.set_column("B:I", 16)


def _write_notebook(root: Path) -> None:
    notebook_path = root / "notebooks" / "claim_risk_analysis.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Insurance Claim Risk Analysis\n\n"
            "This executed notebook inspects the actual artifacts generated by `python main.py`. "
            "The main pipeline performs loading, cleaning, feature preparation, cross-validation, "
            "evaluation, and prediction with PySpark; this notebook presents the saved outputs "
            "without training the model a second time."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "OUTPUTS = ROOT / 'outputs'\n"
            "POWERBI = ROOT / 'dashboard' / 'powerbi_data'\n"
            "RUN = json.loads((OUTPUTS / 'run_metadata.json').read_text(encoding='utf-8'))\n"
            "print(f\"Source rows: {RUN['source_rows']:,}; cleaned rows: {RUN['cleaned_rows']:,}\")\n"
            "print('Risk counts:', RUN['risk_counts'])\n"
            "print('Mappings:', RUN['mappings'])"
        ),
        nbformat.v4.new_markdown_cell(
            "## Cleaning and constructed target\n\n"
            "Question marks/blanks are normalized, duplicates are removed, numeric fields are "
            "typed and imputed, claim amount outliers are capped using the IQR rule, and the "
            "risk score is mapped into Low, Medium, and High using the observed p40/p75 cutoffs."
        ),
        nbformat.v4.new_code_cell(
            "predictions = pd.read_csv(ROOT / 'data' / 'processed' / 'cleaned_claims_with_predictions.csv')\n"
            "print('Prediction table shape:', predictions.shape)\n"
            "print('Null cells:', int(predictions.isna().sum().sum()))\n"
            "display(predictions[['risk_level', 'predicted_risk_level']].head())\n"
            "display(predictions['risk_level'].value_counts().reindex(['Low', 'Medium', 'High']).rename('count'))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Model evaluation\n\n"
            "The majority-class baseline and untuned tree are compared with the best model "
            "selected using three-fold cross-validation. F1 on training versus held-out test "
            "data provides a basic overfitting check."
        ),
        nbformat.v4.new_code_cell(
            "metrics = pd.read_csv(OUTPUTS / 'metrics.csv')\n"
            "per_class = pd.read_csv(OUTPUTS / 'per_class_metrics.csv')\n"
            "confusion = pd.read_csv(OUTPUTS / 'confusion_matrix.csv', index_col=0)\n"
            "display(metrics)\n"
            "display(per_class)\n"
            "display(confusion)\n"
            "print('Best parameters:', RUN['best_params'])\n"
            "print('Train-test F1 gap:', round(RUN['overfit_gap'], 4))\n"
            "print(RUN['overfit_comment'])\n"
            "print(RUN['weighted_precision_note'])"
        ),
        nbformat.v4.new_markdown_cell(
            "## Feature importance and risk distributions\n\n"
            "Feature names come from Spark VectorAssembler attribute metadata. Charts were "
            "created from Spark aggregates and saved as PNG files for the workbook and report."
        ),
        nbformat.v4.new_code_cell(
            "importance = pd.read_csv(OUTPUTS / 'feature_importance.csv')\n"
            "display(importance.head(15))\n"
            "display(Image(filename=str(ROOT / 'dashboard' / 'dashboard_preview.png')))\n"
            "print('Excel dashboard:', ROOT / 'dashboard' / 'Insurance_Claim_Risk_Dashboard.xlsx')"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation and limitations\n\n"
            "The target is a heuristic derived from claim amount, incident severity, and the "
            "fraud indicator, not independently verified future risk. These target-derived "
            "fields are explicitly excluded from model features. `policy_type` is a proxy "
            "derived from `policy_csl`; the held-out random split is not external or temporal "
            "validation. See `README.md` and the generated DOCX report for full details."
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbformat.write(notebook, notebook_path)
    python_scripts = str(Path(sys.executable).parent)
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = python_scripts + os.pathsep + current_path
    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    ).execute()
    nbformat.write(executed, notebook_path)


def _write_docx(root: Path, run: Dict[str, Any]) -> None:
    document = Document()
    document.add_heading("Insurance Claim Risk Analysis", 0)
    document.add_paragraph("Project report generated from the actual PySpark pipeline execution.")
    document.add_heading("1. Introduction", level=1)
    document.add_paragraph(
        "This project builds an exploratory insurance-claim risk classification workflow. "
        "It combines Spark-based data preparation and decision-tree modelling with a "
        "reproducible evaluation and reporting pipeline."
    )
    document.add_heading("2. Problem statement", level=1)
    document.add_paragraph(
        "The objective is to classify claims into rule-derived Low, Medium, or High risk "
        "levels and summarize the model's agreement with those constructed labels. This is "
        "not a validated fraud detector or a production underwriting recommendation."
    )
    document.add_heading("3. Dataset used", level=1)
    document.add_paragraph(
        f"Source: Data/insurance_claims.csv. Actual rows: {run['source_rows']:,}; "
        f"cleaned rows: {run['cleaned_rows']:,}. The source contained claim, policy, "
        "insured, and incident fields. The source does not contain a native policy-type field."
    )
    document.add_paragraph(
        "Mappings: claim_type = incident_type; customer_region = policy_state; "
        "policy_type = policy_csl (coverage-limit tier proxy)."
    )
    document.add_heading("4. Methodology and pipeline", level=1)
    diagram = root / "outputs" / "pipeline_diagram.png"
    document.add_picture(str(diagram), width=Inches(6.6))
    document.add_paragraph(
        "The workflow loads and inspects the authentic CSV, normalizes missing values, "
        "removes duplicates, caps claim amount outliers, derives the target, builds a "
        "leakage-safe Spark ML pipeline, tunes a decision tree with three-fold CV, evaluates "
        "on the held-out split, then exports tables, figures, and reports."
    )
    document.add_heading("5. Risk-label rule and leakage controls", level=1)
    document.add_paragraph(
        "Risk score = 0.5 × normalized total claim amount + 0.3 × normalized incident "
        "severity rank + 0.2 × fraud indicator (1 for Y, else 0). Low is at or below the "
        "empirical p40, Medium is above p40 through p75, and High is above p75. Actual "
        f"cutoffs: p40={run['risk_score_quantiles']['p40']:.6f}, "
        f"p75={run['risk_score_quantiles']['p75']:.6f}."
    )
    document.add_paragraph(
        "Model features exclude total_claim_amount, injury_claim, property_claim, vehicle_claim, "
        "incident_severity, fraud_reported, risk_score, risk_level, policy_number, insured_zip, "
        "and incident_location. An explicit assertion checks this exclusion set."
    )
    document.add_heading("6. Model and actual results", level=1)
    document.add_paragraph(
        f"Selected parameters: {json.dumps(run['best_params'], sort_keys=True)}. "
        "Tuning criterion: weighted F1 averaged over three cross-validation folds."
    )
    _table(document, run["metrics_frame"])
    document.add_heading("Per-class held-out metrics (Low / Medium / High)", level=2)
    _table(document, run["per_class"])
    document.add_heading("Actual-by-predicted confusion matrix", level=2)
    _table(document, run["confusion_matrix"].reset_index())
    document.add_paragraph(run["weighted_precision_note"])
    document.add_paragraph(
        f"Tuned train weighted F1={run['train_metrics']['weighted_f1']:.4f}; "
        f"test weighted F1={run['metrics']['weighted_f1']:.4f}; "
        f"train-minus-test gap={run['overfit_gap']:.4f}. {run['overfit_comment']}"
    )
    if run["metrics"]["accuracy"] > 0.90:
        document.add_paragraph(
            "LEAKAGE AUDIT: held-out accuracy exceeds 0.90. The explicit exclusion assertion "
            "passed, but this high score must be investigated before interpretation."
        )
    document.add_heading("7. Captured code and actual output", level=1)
    code = (
        "results = train_evaluate(spark, cleaned, cleaning_metadata, outputs_dir)\n"
        "print(f\"Best CV parameters: {results['best_params']}\")\n"
        "print(f\"Tuned held-out metrics: {results['metrics']}\")"
    )
    code_paragraph = document.add_paragraph()
    code_run = code_paragraph.add_run(code)
    code_run.font.name = "Consolas"
    code_run.font.size = Pt(8)
    document.add_paragraph("Actual output excerpt generated from this completed run:")
    output = (
        f"Best params: {json.dumps(run['best_params'], sort_keys=True)}\n"
        f"Majority baseline: {run['metrics_frame'].set_index('model').loc['majority_class_baseline'].to_dict()}\n"
        f"Baseline test metrics: {run['metrics_frame'].set_index('model').loc['untuned_tree'].to_dict()}\n"
        f"Tuned test metrics: {run['metrics_frame'].set_index('model').loc['tuned_tree'].to_dict()}\n"
        f"Per-class:\n{run['per_class'].to_string(index=False)}\n"
        f"Confusion matrix:\n{run['confusion_matrix'].to_string()}"
    )
    paragraph = document.add_paragraph()
    run_text = paragraph.add_run(output)
    run_text.font.name = "Consolas"
    run_text.font.size = Pt(8)
    document.add_heading("8. Dashboard and visual findings", level=1)
    document.add_paragraph(
        "Power BI Desktop screenshot placeholder (capture the user-built report here): "
        "[Insert Power BI Desktop screenshot after opening the supplied CSVs.]"
    )
    document.add_picture(str(root / "dashboard" / "dashboard_preview.png"), width=Inches(6.2))
    document.add_paragraph(
        "The dashboard includes KPI cards, native Excel charts, risk-by-category summaries, "
        "and cleaned records with predicted risk. Companion charts include grouped counts, "
        "100% stacked risk share, policy/region/claim heatmaps, and the overall class distribution."
    )
    document.add_heading("9. Conclusion, insights, and future work", level=1)
    document.add_paragraph(
        f"The tuned held-out accuracy was {run['metrics']['accuracy']:.4f} and weighted F1 was "
        f"{run['metrics']['weighted_f1']:.4f}. {run['overfit_comment']} Future work should "
        "validate labels against an independently defined outcome, use time-aware or external "
        "validation, compare multiple models, calibrate probabilities, and confirm policy-type "
        "semantics with domain stakeholders."
    )
    document.add_heading("10. Limitations", level=1)
    document.add_paragraph(
        "The target is heuristic and reuses claim/fraud fields only to derive the label; those "
        "fields are excluded from model inputs. The 1,000-row data and random holdout do not "
        "establish external generalization. `policy_csl` is a proxy for policy type. Model "
        f"save status: {'successful' if not run['model_save_error'] else 'failed non-fatally: ' + run['model_save_error']}."
    )
    document.add_heading("11. References", level=1)
    for reference in (
        "Provided authentic source: Data/insurance_claims.csv.",
        "Apache Spark 3.5 documentation: https://spark.apache.org/docs/3.5.7/",
        "PySpark ML guide: https://spark.apache.org/docs/3.5.7/ml-guide.html",
        "pandas documentation: https://pandas.pydata.org/docs/",
        "XlsxWriter documentation: https://xlsxwriter.readthedocs.io/",
    ):
        document.add_paragraph(reference, style="List Bullet")
    document.add_heading("12. Viva questions and answers", level=1)
    questions = [
        ("What does the model predict?", "A three-level heuristic claim-risk label: Low, Medium, or High."),
        ("How is risk score constructed?", "A weighted sum of normalized claim amount, severity rank, and fraud indicator."),
        ("Why exclude claim amount and fraud from model features?", "They are ingredients used to define the target, so including them would leak target information."),
        ("Why is policy_type a proxy?", "The source has no native policy_type field; it is mapped from policy_csl, the coverage-limit tier."),
        ("What is the CV setup?", "Three-fold cross-validation over the decision-tree parameter grid; weighted F1 is the selection metric."),
        ("What does weighted precision mean?", "Per-class precision averaged with actual class support as weights."),
        ("How do you assess overfitting?", "Compare training and held-out test weighted F1; a large positive gap suggests overfitting."),
        ("What is the purpose of the confusion matrix?", "It shows correct and incorrect predictions for each actual/predicted class pair."),
        ("Why might model saving fail on Windows?", "Spark's local Hadoop filesystem may need winutils.exe; model persistence is optional and handled non-fatally."),
        ("What is the main limitation?", "The outcome labels are constructed heuristically and are not independently validated ground truth."),
    ]
    for question, answer in questions:
        document.add_paragraph(question, style="List Number")
        document.add_paragraph(answer)
    report_path = root / "report" / "Insurance_Claim_Risk_Report.docx"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(report_path)


def _add_slide(
    presentation: Presentation,
    title: str,
    body: str,
    notes: str,
    image: Path | None = None,
) -> None:
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    title_box = slide.shapes.add_textbox(PptInches(0.55), PptInches(0.25), PptInches(12.1), PptInches(0.6))
    title_frame = title_box.text_frame
    title_frame.text = title
    title_run = title_frame.paragraphs[0].runs[0]
    title_run.font.size = PptPt(27)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(23, 50, 77)
    if image and image.is_file():
        slide.shapes.add_picture(str(image), PptInches(0.65), PptInches(1.05), width=PptInches(7.7))
        left, top, width, height = PptInches(8.55), PptInches(1.05), PptInches(4.2), PptInches(5.8)
    else:
        left, top, width, height = PptInches(0.8), PptInches(1.1), PptInches(11.9), PptInches(5.9)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    frame = textbox.text_frame
    frame.word_wrap = True
    frame.text = body
    for paragraph in frame.paragraphs:
        paragraph.font.size = PptPt(18)
        paragraph.space_after = PptPt(10)
    slide.notes_slide.notes_text_frame.text = notes


def _write_presentation(root: Path, run: Dict[str, Any]) -> None:
    presentation = Presentation()
    presentation.slide_width = PptInches(13.333)
    presentation.slide_height = PptInches(7.5)
    metric_table = run["metrics_frame"].set_index("model")
    baseline = metric_table.loc["untuned_tree"]
    tuned = metric_table.loc["tuned_tree"]
    majority = metric_table.loc["majority_class_baseline"]
    _add_slide(
        presentation,
        "Insurance Claim Risk Analysis",
        "PySpark decision-tree workflow\nActual dataset run • model evaluation • dashboard",
        f"Introduce the project and explain the key limitation: risk labels are rule-derived. Actual cleaned rows: {run['cleaned_rows']}.",
    )
    _add_slide(
        presentation,
        "Problem and dataset",
        f"• Source: authentic insurance claims CSV\n• {run['source_rows']:,} source rows; {run['cleaned_rows']:,} cleaned rows\n"
        "• Target: Low / Medium / High risk\n• Mappings: claim_type ← incident_type; customer_region ← policy_state; policy_type ← policy_csl proxy",
        "Describe source fields and state that policy_type is not native to the source dataset.",
    )
    _add_slide(
        presentation,
        "Methodology and processing pipeline",
        "Spark CSV ingestion → inspection → missing-value normalization → deduplication / IQR cap → risk-label derivation → feature pipeline → decision tree / CV → evaluation and exports",
        "Walk through the diagram from raw CSV to deliverables. Spark performs processing and modeling; pandas writes small CSV exports.",
        root / "outputs" / "pipeline_diagram.png",
    )
    _add_slide(
        presentation,
        "Risk rule and leakage controls",
        f"Risk score = 0.5 × normalized claim amount + 0.3 × normalized severity + 0.2 × fraud indicator\n"
        f"Cutoffs: p40={run['risk_score_quantiles']['p40']:.4f}; p75={run['risk_score_quantiles']['p75']:.4f}\n"
        f"Excluded model fields: {', '.join(sorted(LEAKAGE_COLUMNS))}",
        "Explain that target ingredients and identifiers are excluded from feature inputs. This rule-derived label is not ground truth.",
    )
    _add_slide(
        presentation,
        "Training and cross-validation",
        f"Decision tree • three-fold CV • weighted F1 selection\nBest parameters: {json.dumps(run['best_params'], sort_keys=True)}\n"
        f"Code: results = train_evaluate(spark, cleaned, cleaning_metadata, outputs_dir)\n"
        f"Feature columns: {len(run['feature_columns'])}\nClass weighting used: {run['use_weights']}",
        "Explain train-only class weight estimation and the selected tree hyperparameters.",
    )
    _add_slide(
        presentation,
        "Model comparison and generalization",
        f"Majority baseline: accuracy {majority['accuracy']:.3f}, weighted F1 {majority['weighted_f1']:.3f}\n"
        f"Untuned tree: accuracy {baseline['accuracy']:.3f}, weighted F1 {baseline['weighted_f1']:.3f}\n"
        f"Tuned tree: accuracy {tuned['accuracy']:.3f}, weighted F1 {tuned['weighted_f1']:.3f}\n"
        f"Tuned train/test F1: {run['train_metrics']['weighted_f1']:.3f} / {tuned['weighted_f1']:.3f}\n"
        f"Train-test gap: {run['overfit_gap']:.3f}. {run['overfit_comment']}",
        "Compare model performance and use the train-test F1 gap as a basic overfitting check. These are estimates on the fixed random holdout.",
    )
    _add_slide(
        presentation,
        "Per-class evaluation and confusion matrix",
        run["per_class"].to_string(index=False, float_format=lambda value: f"{value:.3f}")
        + "\n\nConfusion matrix (rows actual, columns predicted):\n"
        + run["confusion_matrix"].to_string()
        + "\n\n"
        + run["weighted_precision_note"],
        "Discuss the class-level results and the confusion matrix. Weighted precision averages class precision by support; false negatives reduce recall.",
    )
    _add_slide(
        presentation,
        "Feature importance",
        run["feature_importance"].head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "Read feature names from Spark VectorAssembler metadata. Explain that tree importance is model-specific and not causal.",
        root / "outputs" / "figures" / "feature_importance.png",
    )
    _add_slide(
        presentation,
        "Dashboard and risk distributions",
        "Actual run dashboard preview\nGrouped counts • 100% stacked risk composition • policy / region / claim-type heatmaps • overall distribution",
        "Use the preview to orient the audience, then point to the Excel workbook and Power BI CSV exports.",
        root / "dashboard" / "dashboard_preview.png",
    )
    _add_slide(
        presentation,
        "Conclusions, limitations, references",
        f"Tuned test accuracy {tuned['accuracy']:.3f}; weighted F1 {tuned['weighted_f1']:.3f}.\n"
        "Limitations: rule-derived labels, 1,000 claims, random holdout, proxy policy_type, no external validation.\n"
        "Next steps: validate target independently; temporal/external testing; compare calibrated models.\n"
        "References: supplied CSV; Apache Spark 3.5 docs; PySpark ML guide; pandas / XlsxWriter docs.",
        "Summarize actual results, emphasize the target's heuristic nature, and propose independent validation.",
    )
    questions = [
        ("What is predicted?", "A three-level rule-derived risk label."),
        ("How is the score built?", "Normalized claim amount, severity, and fraud indicator with weights 0.5/0.3/0.2."),
        ("Why exclude target ingredients?", "To prevent direct target leakage."),
        ("What does policy_type represent?", "A coverage-limit proxy mapped from policy_csl."),
        ("How was the tree selected?", "Three-fold CV optimizing weighted F1."),
        ("What does weighted precision mean?", "Support-weighted average of class precision."),
        ("How is overfitting checked?", "Training versus held-out test weighted F1 gap."),
        ("What does confusion matrix show?", "Actual-versus-predicted class counts."),
        ("Why might saving fail on Windows?", "Hadoop local permissions may require winutils.exe."),
        ("What is the primary limitation?", "The heuristic label is not independently validated ground truth."),
    ]
    for offset in (0, 5):
        body = "\n\n".join(
            f"Q{index + 1}. {question}\nA. {answer}"
            for index, (question, answer) in enumerate(questions[offset : offset + 5], start=offset)
        )
        _add_slide(
            presentation,
            f"Viva questions and answers ({offset + 1}–{offset + 5})",
            body,
            "Use the questions as a viva preparation appendix; answers are concise and based on the implemented project.",
        )
    report_path = root / "report" / "presentation.pptx"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(report_path)


def create_project_deliverables(root: Path, run: Dict[str, Any]) -> None:
    """Generate the README, notebook, workbook, Power BI guide, and reports."""
    create_pipeline_diagram(root / "outputs" / "pipeline_diagram.png")
    _write_powerbi_guide(root)
    _write_excel_dashboard(root, run)
    _write_readme(root, run)
    _write_docx(root, run)
    _write_presentation(root, run)
    _write_notebook(root)
