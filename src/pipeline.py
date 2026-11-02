"""Build, tune, evaluate, and persist the PySpark claim-risk classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _weight_training_rows(train: DataFrame, test: DataFrame) -> Tuple[DataFrame, DataFrame, bool]:
    """Apply inverse-frequency class weights based on the training partition only."""
    counts = {row["risk_level"]: row["count"] for row in train.groupBy("risk_level").count().collect()}
    if not counts:
        raise ValueError("No training labels are available.")
    smallest = min(counts.values())
    imbalanced = smallest > 0 and max(counts.values()) / smallest > 1.5
    weights = {label: (sum(counts.values()) / (len(counts) * count)) for label, count in counts.items()}
    weight_expression = F.create_map(
        *[item for label, value in weights.items() for item in (F.lit(label), F.lit(float(value)))]
    )[F.col("risk_level")]
    return (
        train.withColumn("_class_weight", F.coalesce(weight_expression, F.lit(1.0))),
        test.withColumn("_class_weight", F.coalesce(weight_expression, F.lit(1.0))),
        imbalanced,
    )


def _make_pipeline(numeric_features: List[str], categorical_features: List[str], weighted: bool) -> Tuple[Pipeline, DecisionTreeClassifier]:
    """Create one leakage-safe Spark pipeline with all learned transformations."""
    stages = []
    imputed_columns = []
    if numeric_features:
        imputed_columns = [f"_imputed_{name}" for name in numeric_features]
        stages.append(
            Imputer(
                strategy="median",
                inputCols=numeric_features,
                outputCols=imputed_columns,
            )
        )
    index_columns = [f"_index_{name}" for name in categorical_features]
    if categorical_features:
        stages.extend(
            [
                StringIndexer(
                    inputCol=source,
                    outputCol=output,
                    handleInvalid="keep",
                )
                for source, output in zip(categorical_features, index_columns)
            ]
        )
        stages.append(
            OneHotEncoder(
                inputCols=index_columns,
                outputCols=[f"_ohe_{name}" for name in categorical_features],
                handleInvalid="keep",
                dropLast=False,
            )
        )
    feature_inputs = imputed_columns + [f"_ohe_{name}" for name in categorical_features]
    stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="_raw_features", handleInvalid="keep"))
    stages.append(MinMaxScaler(inputCol="_raw_features", outputCol="features"))
    stages.append(StringIndexer(inputCol="risk_level", outputCol="label", handleInvalid="keep"))
    classifier = DecisionTreeClassifier(
        labelCol="label",
        featuresCol="features",
        seed=42,
        weightCol="_class_weight" if weighted else None,
    )
    stages.append(classifier)
    return Pipeline(stages=stages), classifier


def _label_names(model: PipelineModel) -> List[str]:
    """Return the fitted label index mapping."""
    for stage in model.stages:
        if (
            (isinstance(stage, StringIndexer) or stage.__class__.__name__ == "StringIndexerModel")
            and stage.getInputCol() == "risk_level"
            and hasattr(stage, "labels")
        ):
            return list(stage.labels)
    raise ValueError("Fitted pipeline did not contain a label mapping.")


def _metrics(predictions: DataFrame, label_names: List[str]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute overall and per-class metrics from Spark predictions."""
    evaluators = {
        "accuracy": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        ),
        "weighted_precision": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
        ),
        "weighted_recall": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedRecall"
        ),
        "weighted_f1": MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        ),
    }
    summary = {name: float(evaluator.evaluate(predictions)) for name, evaluator in evaluators.items()}
    class_rows = []
    for index, name in enumerate(label_names):
        counts = predictions.agg(
            F.sum(F.when((F.col("label") == index) & (F.col("prediction") == index), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("label") == index) & (F.col("prediction") != index), 1).otherwise(0)).alias("fn"),
            F.sum(F.when((F.col("label") != index) & (F.col("prediction") == index), 1).otherwise(0)).alias("fp"),
            F.sum(F.when(F.col("label") == index, 1).otherwise(0)).alias("support"),
        ).first()
        tp, fn, fp = int(counts["tp"] or 0), int(counts["fn"] or 0), int(counts["fp"] or 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(counts["support"] or 0),
            }
        )
    return summary, pd.DataFrame(class_rows)


def _confusion_matrix(predictions: DataFrame, labels: List[str]) -> pd.DataFrame:
    """Build an actual-by-predicted confusion matrix with readable class labels."""
    rows = (
        predictions.groupBy("label", "prediction")
        .count()
        .collect()
    )
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for row in rows:
        actual, predicted = int(row["label"]), int(row["prediction"])
        if actual < len(labels) and predicted < len(labels):
            matrix.loc[labels[actual], labels[predicted]] = int(row["count"])
    matrix.index.name = "actual"
    matrix.columns.name = "predicted"
    return matrix


def _feature_importance(model: PipelineModel) -> pd.DataFrame:
    """Map tree importance indices to assembled real-world feature names."""
    tree = model.stages[-1]
    metadata = model.transformSchema if False else None
    del metadata
    feature_metadata = None
    for stage in model.stages:
        if stage.__class__.__name__ == "VectorAssembler":
            feature_metadata = stage
    del feature_metadata
    importance = tree.featureImportances
    # The fitted assembler output metadata lives on the transformed training column.
    # Build names from indexer labels where possible and fall back to stable source prefixes.
    metadata_names = getattr(model, "_risk_feature_names", None)
    names = metadata_names or [f"feature_{i}" for i in range(importance.size)]
    return (
        pd.DataFrame(
            {"feature": names[: importance.size], "importance": importance.toArray()[: len(names)]}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _feature_names(predictions: DataFrame) -> List[str]:
    """Read feature attribute names emitted by VectorAssembler metadata."""
    metadata = predictions.schema["_raw_features"].metadata.get("ml_attr", {})
    attributes = metadata.get("attrs", {})
    pairs = []
    for group in attributes.values():
        for attribute in group:
            pairs.append((int(attribute["idx"]), attribute["name"]))
    pairs.sort(key=lambda pair: pair[0])
    return [name for _, name in pairs]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def train_evaluate(
    spark: SparkSession,
    cleaned: DataFrame,
    metadata: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    """Fit baseline and cross-validated trees, evaluate, save, reload, and report results."""
    print("\n========== TASK 3: ENCODE FEATURES ==========")
    print("\n========== TASK 4: TRAIN AND TUNE DECISION TREE ==========")
    print("Leakage exclusions: claim amounts, severity, fraud, risk_score, risk_level, and ID-like fields.")
    feature_cols = list(metadata["numeric_model_features"]) + list(metadata["categorical_model_features"])
    excluded = {
        "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
        "incident_severity", "fraud_reported", "risk_score", "risk_level",
        "policy_number", "insured_zip", "incident_location",
    }
    if set(feature_cols) & excluded:
        raise AssertionError(f"Leakage or identifier feature detected: {set(feature_cols) & excluded}")
    print(f"Numeric feature count: {len(metadata['numeric_model_features'])}")
    print(f"Categorical feature count: {len(metadata['categorical_model_features'])}")
    train, test = cleaned.randomSplit([0.8, 0.2], seed=42)
    train, test, use_weights = _weight_training_rows(train, test)
    print("Training label distribution:")
    train.groupBy("risk_level").count().orderBy("risk_level").show()
    print("Test label distribution:")
    test.groupBy("risk_level").count().orderBy("risk_level").show()
    train = train.cache()
    train.count()
    pipeline, classifier = _make_pipeline(
        metadata["numeric_model_features"],
        metadata["categorical_model_features"],
        use_weights,
    )
    baseline_estimator = DecisionTreeClassifier(
        labelCol="label",
        featuresCol="features",
        maxDepth=5,
        maxBins=32,
        minInstancesPerNode=1,
        seed=42,
        weightCol="_class_weight" if use_weights else None,
    )
    baseline_pipeline, _ = _make_pipeline(
        metadata["numeric_model_features"],
        metadata["categorical_model_features"],
        use_weights,
    )
    baseline_pipeline = Pipeline(stages=baseline_pipeline.getStages()[:-1] + [baseline_estimator])
    baseline_model = baseline_pipeline.fit(train)
    baseline_label_names = _label_names(baseline_model)
    baseline_predictions = baseline_model.transform(test)
    baseline_metrics, baseline_per_class = _metrics(baseline_predictions, baseline_label_names)
    baseline_per_class["class"] = pd.Categorical(
        baseline_per_class["class"], categories=["Low", "Medium", "High"], ordered=True
    )
    baseline_per_class = baseline_per_class.sort_values("class").reset_index(drop=True)
    baseline_train_metrics, _ = _metrics(baseline_model.transform(train), baseline_label_names)
    print(f"Untuned tree test metrics: {baseline_metrics}")

    # Keep the assignment's tuning step practical for the 1,000-row dataset and
    # Windows laptops: three folds across 16 configurations launches 48 full
    # Spark pipeline fits and can take several minutes on local mode.
    grid = (
        ParamGridBuilder()
        .addGrid(classifier.maxDepth, [3, 5, 7])
        .addGrid(classifier.maxBins, [32])
        .addGrid(classifier.minInstancesPerNode, [1, 5])
        .build()
    )
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=2,
        seed=42,
        parallelism=1,
    )
    cv_model = validator.fit(train)
    cv_results = pd.DataFrame(
        [
            {
                "maxDepth": int(param_map[classifier.maxDepth]),
                "maxBins": int(param_map[classifier.maxBins]),
                "minInstancesPerNode": int(param_map[classifier.minInstancesPerNode]),
                "mean_cv_weighted_f1": float(score),
            }
            for param_map, score in zip(grid, cv_model.avgMetrics)
        ]
    ).sort_values("mean_cv_weighted_f1", ascending=False).reset_index(drop=True)
    best_model = cv_model.bestModel
    label_names = _label_names(best_model)
    print(f"Stored label mapping (index order): {label_names}")
    test_predictions = best_model.transform(test).cache()
    test_metrics, per_class = _metrics(test_predictions, label_names)
    train_metrics, _ = _metrics(best_model.transform(train), label_names)
    class_order = ["Low", "Medium", "High"]
    per_class["class"] = pd.Categorical(per_class["class"], categories=class_order, ordered=True)
    per_class = per_class.sort_values("class").reset_index(drop=True)
    confusion = _confusion_matrix(test_predictions, label_names)
    confusion = confusion.reindex(index=class_order, columns=class_order, fill_value=0)
    majority_label = train.groupBy("risk_level").count().orderBy(F.desc("count")).first()["risk_level"]
    majority_accuracy = test.where(F.col("risk_level") == majority_label).count() / test.count()
    majority_precision = float(majority_accuracy)
    majority_recall = 1.0
    majority_f1 = 2 * majority_precision * majority_recall / (majority_precision + majority_recall)
    majority_baseline = {
        "majority_class": majority_label,
        "accuracy": float(majority_accuracy),
        "weighted_precision": float(majority_accuracy * majority_precision),
        "weighted_recall": float(majority_accuracy),
        "weighted_f1": float(majority_accuracy * majority_f1),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_pipeline_model"
    label_array = F.array(*[F.lit(label) for label in label_names])
    reloaded = None
    model_save_error = None
    try:
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
        best_model.write().overwrite().save(str(model_path))
        reloaded = PipelineModel.load(str(model_path))
        demo = reloaded.transform(test.limit(5))
        demo = demo.withColumn(
            "predicted_risk_level", F.element_at(label_array, F.col("prediction").cast("int") + 1)
        ).withColumn("probability", F.col("probability").cast("string"))
        print("Reloaded saved pipeline sample predictions (label + probability):")
        demo.select("risk_level", "predicted_risk_level", "probability").show(truncate=False)
    except Exception as exc:
        error_text = str(exc)
        if "HADOOP_HOME" in error_text or "winutils" in error_text.lower():
            reason = "Windows Hadoop local filesystem could not find winutils.exe (HADOOP_HOME is unset or incomplete)."
        else:
            reason = " ".join(error_text.splitlines())[:500]
        model_save_error = f"{type(exc).__name__}: {reason}"
        print(f"WARNING: model persistence/reload failed; other outputs will still be generated. {model_save_error}")
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)

    # Attach the real input attribute labels before mapping tree importance indices.
    names = _feature_names(best_model.transform(test.limit(1)))
    importances = _feature_importance(best_model)
    if names and len(names) == len(importances):
        importances["feature"] = names
        importances = importances.sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        raise ValueError(
            f"Could not map all {len(importances)} tree importance values to real input feature names "
            f"(found {len(names)} names)."
        )
    importances.to_csv(output_dir / "feature_importance.csv", index=False)
    importances.head(15).to_csv(output_dir / "top_15_feature_importance.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv", index=True)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    baseline_per_class.to_csv(output_dir / "baseline_per_class_metrics.csv", index=False)
    cv_results.to_csv(output_dir / "cross_validation_results.csv", index=False)
    metrics_rows = [
        {"model": "majority_class_baseline", **majority_baseline},
        {"model": "untuned_tree", **baseline_metrics},
        {"model": "tuned_tree", **test_metrics},
        {"model": "untuned_tree_train", **baseline_train_metrics},
        {"model": "tuned_tree_train", **train_metrics},
    ]
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    best_tree = best_model.stages[-1]
    best_params = {
        "maxDepth": int(best_tree.getMaxDepth()),
        "maxBins": int(best_tree.getMaxBins()),
        "minInstancesPerNode": int(best_tree.getMinInstancesPerNode()),
    }
    lower_recall = baseline_per_class[
        baseline_per_class["recall"] < baseline_per_class["precision"]
    ]
    class_recall_note = "; ".join(
        f"{row['class']}: precision={row['precision']:.3f}, recall={row['recall']:.3f}"
        for _, row in lower_recall.iterrows()
    )
    weighted_precision_note = (
        f"Baseline weighted precision={baseline_metrics['weighted_precision']:.4f} exceeds "
        f"weighted recall={baseline_metrics['weighted_recall']:.4f}. "
        "Precision is the share of predictions that are correct; recall is the share of "
        "actual cases found. The per-class results where precision exceeds recall are: "
        f"{class_recall_note or 'none'}; this indicates missed actual cases (false negatives) "
        "are lowering recall more than false positives lower precision. The weighted scores "
        "are support-weighted, so interpret them with the per-class metrics and confusion matrix."
    )
    overfit_gap = train_metrics["weighted_f1"] - test_metrics["weighted_f1"]
    text = [
        "Insurance Claim Risk Analysis - actual test-set results",
        f"Label index mapping: {json.dumps(label_names)}",
        f"Best cross-validation parameters (3 folds, weighted F1): {json.dumps(best_params)}",
        f"Majority-class baseline: {json.dumps(majority_baseline)}",
        f"Untuned tree test: {json.dumps(baseline_metrics)}",
        f"Untuned tree train: {json.dumps(baseline_train_metrics)}",
        f"Tuned tree test: {json.dumps(test_metrics)}",
        f"Tuned tree train: {json.dumps(train_metrics)}",
        f"Baseline per-class metrics:\n{baseline_per_class.to_string(index=False)}",
        f"Tuned tree train-test weighted F1 gap: {overfit_gap:.4f}",
        f"Weighted precision vs recall: {weighted_precision_note}",
        f"Per-class metrics:\n{per_class.to_string(index=False)}",
        f"Confusion matrix (rows=actual, columns=predicted):\n{confusion.to_string()}",
        f"Top feature importances:\n{importances.head(15).to_string(index=False)}",
        f"Selected tree parameters: {json.dumps(best_params)}",
        "Interpretation: train/test F1 gaps indicate the extent of possible overfitting; "
        "all figures are estimates on the held-out split.",
    ]
    _write_text(output_dir / "classification_report.txt", "\n\n".join(text))
    try:
        tree_text = best_model.stages[-1].toDebugString
    except Exception as exc:
        tree_text = f"Unable to render tree text: {exc}"
    _write_text(output_dir / "decision_tree.txt", tree_text)
    print("\n========== TASK 5: TEST-SET EVALUATION ==========")
    print(f"Majority-class baseline: {majority_baseline}")
    print(f"Tuned tree test metrics: {test_metrics}")
    print(f"Tuned tree train F1={train_metrics['weighted_f1']:.4f}; "
          f"test F1={test_metrics['weighted_f1']:.4f}")
    print(f"Best cross-validation parameters: {best_params}")
    print(f"Weighted precision note: {weighted_precision_note}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion.to_string())
    print("Per-class metrics:")
    print(per_class.to_string(index=False))
    print("Top feature importances:")
    print(importances.head(15).to_string(index=False))
    if test_metrics["accuracy"] > 0.90:
        print("LEAKAGE AUDIT WARNING: test accuracy exceeds 90%; review the explicit exclusion checks.")

    export = best_model.transform(cleaned).withColumn(
        "predicted_risk_level", F.element_at(label_array, F.col("prediction").cast("int") + 1)
    ).withColumn("probability", F.col("probability").cast("string"))
    predictions_path = output_dir.parent / "data" / "processed" / "cleaned_claims_with_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    export.select(*cleaned.columns, "predicted_risk_level", "probability").toPandas().to_csv(
        predictions_path, index=False
    )
    return {
        "model": best_model,
        "reloaded_model": reloaded,
        "model_save_error": model_save_error,
        "train": train,
        "test": test,
        "test_predictions": test_predictions,
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_per_class": baseline_per_class,
        "baseline_train_metrics": baseline_train_metrics,
        "majority_baseline": majority_baseline,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "feature_importance": importances,
        "label_mapping": label_names,
        "best_params": best_params,
        "cv_results": cv_results,
        "overfit_gap": overfit_gap,
        "weighted_precision_note": weighted_precision_note,
        "use_weights": use_weights,
        "feature_columns": feature_cols,
    }
