# Insurance Claim Risk Analysis

## Project status and actual run

The workflow ran against the authentic source file `Data/insurance_claims.csv` with
1,000 source rows and 1,000 cleaned rows. The model
features intentionally exclude the derived target ingredients and identifier-like columns.
The best cross-validation tree parameters were `{"maxBins": 32, "maxDepth": 5, "minInstancesPerNode": 5}`.

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Majority-class baseline (Low) | 0.4074 | 0.1660 | 0.4074 | 0.2359 |
| Untuned tree | 0.5741 | 0.7087 | 0.5741 | 0.5622 |
| Tuned tree | 0.5556 | 0.6581 | 0.5556 | 0.5370 |

Tuned-tree weighted F1 was 0.5481 on training data and
0.5370 on the held-out test partition (train minus test:
0.0111). The train score is higher by 0.0111; this is a modest train/test gap.

### Per-class held-out metrics

 class  precision  recall     f1  support
   Low     0.9118  0.4697 0.6200       66
Medium     0.4513  0.8793 0.5965       58
  High     0.5333  0.2105 0.3019       38

### Precision versus recall

Baseline weighted precision=0.7087 exceeds weighted recall=0.5741. Precision is the share of predictions that are correct; recall is the share of actual cases found. The per-class results where precision exceeds recall are: Low: precision=1.000, recall=0.470; High: precision=0.588, recall=0.263; this indicates missed actual cases (false negatives) are lowering recall more than false positives lower precision. The weighted scores are support-weighted, so interpret them with the per-class metrics and confusion matrix.

## Setup and run

1. Use Python 3.11 and install the pinned/project dependencies:
   `python -m pip install -r requirements.txt`
2. Ensure Java and Spark 3.5 are installed and `SPARK_HOME` points to the Spark
   distribution. The project reads the authentic CSV from `Data/insurance_claims.csv`.
3. On Windows, Hadoop's `winutils.exe` may be required for Spark local filesystem
   permissions. If `C:\hadoop\bin\winutils.exe` exists, `main.py` configures
   `HADOOP_HOME=C:\hadoop` when it is unset. If it is absent, the runner prints a warning
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
  through p75, and High above p75. Actual cut points: p40=0.365552,
  p75=0.543031.

## Leakage controls

The model input assertion rejects these fields: `fraud_reported, incident_location, incident_severity, injury_claim, insured_zip, policy_number, property_claim, risk_level, risk_score, total_claim_amount, vehicle_claim`.
These contain the target, target ingredients, claim amounts, or identifiers. The derived
risk label is a heuristic target, not a ground-truth future outcome; model scores measure
agreement with that constructed rule, not proven real-world fraud or loss prediction.
If held-out test accuracy exceeds 0.90, treat it as a leakage signal and investigate the
feature list, transformations, and split before interpreting the result. Actual tuned
test accuracy in this run: 0.5556.

## Model persistence

Not saved in this Windows run: `Py4JJavaError: Windows Hadoop local filesystem could not find winutils.exe (HADOOP_HOME is unset or incomplete).`. This is non-fatal; the metrics, CSVs, reports, and visualizations were still generated.

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
