"""Validation-only diagnostics shared by controlled repair experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def classification(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": (tp + tn) / len(labels),
    }


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, specificity_floor: float = 0.90
) -> dict:
    """Maximize recall subject to a predeclared specificity floor."""
    if not 0.0 < specificity_floor <= 1.0:
        raise ValueError("specificity_floor must be in (0, 1]")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    eligible = [
        classification(labels, probabilities, float(threshold))
        for threshold in candidates
    ]
    eligible = [row for row in eligible if row["specificity"] >= specificity_floor]
    if not eligible:
        raise RuntimeError("no threshold reaches the specificity floor")
    eligible.sort(
        key=lambda row: (row["recall"], row["f1"], row["specificity"], -row["threshold"]),
        reverse=True,
    )
    return eligible[0]


def ranking(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prevalence = float(labels.mean())
    return {
        "windows": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "pr_auc_lift": float(average_precision_score(labels, probabilities) / prevalence),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def attach_manifest_metadata(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    columns = [
        column
        for column in ("event_id", "record_id", "dataset")
        if column not in output.columns
    ]
    if columns:
        metadata = manifest[["window_id", *columns]].drop_duplicates("window_id")
        output = output.merge(metadata, on="window_id", how="left", validate="many_to_one")
    missing_positive_events = (
        output.loc[output["label"].eq(1), "event_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )
    if missing_positive_events.any():
        raise ValueError("positive validation windows are missing event IDs")
    return output


def summarize_predictions(predictions: pd.DataFrame) -> dict:
    required = {
        "window_id", "subject_id", "source_family", "label", "event_id",
        "lead_bin", "negative_stratum", "probability",
    }
    if missing := required - set(predictions.columns):
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["label"] = frame["label"].astype(int)
    frame["probability"] = frame["probability"].astype(float)
    labels = frame.label.to_numpy(dtype=np.uint8)
    probabilities = frame.probability.to_numpy(dtype=np.float64)
    global_operating = select_threshold(labels, probabilities)
    by_source = {}
    for source, group in frame.groupby("source_family", sort=True):
        source_labels = group.label.to_numpy(dtype=np.uint8)
        source_probabilities = group.probability.to_numpy(dtype=np.float64)
        source_operating = select_threshold(source_labels, source_probabilities)
        positive = group[group.label.eq(1)]
        event_maximum = positive.groupby(
            ["subject_id", "event_id"], dropna=False
        ).probability.max()
        detected = int((event_maximum >= source_operating["threshold"]).sum())
        patient_recalls = []
        patient_specificities = []
        for _, patient in group.groupby("subject_id"):
            patient_metrics = classification(
                patient.label.to_numpy(dtype=np.uint8),
                patient.probability.to_numpy(dtype=np.float64),
                source_operating["threshold"],
            )
            if int(patient.label.sum()) > 0:
                patient_recalls.append(patient_metrics["recall"])
            if int(patient.label.eq(0).sum()) > 0:
                patient_specificities.append(patient_metrics["specificity"])
        by_source[str(source)] = {
            "ranking": ranking(source_labels, source_probabilities),
            "global_operating_point": classification(
                source_labels, source_probabilities, global_operating["threshold"]
            ),
            "source_operating_point": source_operating,
            "event_sensitivity_at_source_threshold": {
                "events": int(len(event_maximum)),
                "events_detected": detected,
                "events_missed": int(len(event_maximum) - detected),
                "sensitivity": detected / len(event_maximum) if len(event_maximum) else 0.0,
            },
            "patient_macro_at_source_threshold": {
                "positive_patients": len(patient_recalls),
                "recall_mean": float(np.mean(patient_recalls)),
                "recall_median": float(np.median(patient_recalls)),
                "zero_recall_patients": int(np.sum(np.asarray(patient_recalls) == 0.0)),
                "negative_patients": len(patient_specificities),
                "specificity_mean": float(np.mean(patient_specificities)),
                "specificity_median": float(np.median(patient_specificities)),
            },
        }
    recall_by_lead_bin = {}
    for lead_bin, group in frame[frame.label.eq(1)].groupby("lead_bin", sort=True):
        detected = group.probability.to_numpy() >= global_operating["threshold"]
        recall_by_lead_bin[str(lead_bin)] = {
            "positives": int(len(group)),
            "true_positives": int(detected.sum()),
            "recall": float(detected.mean()),
        }
    return {
        "ranking": ranking(labels, probabilities),
        "global_operating_point": global_operating,
        "by_source": by_source,
        "recall_by_lead_bin": recall_by_lead_bin,
    }


__all__ = [
    "attach_manifest_metadata",
    "classification",
    "ranking",
    "select_threshold",
    "summarize_predictions",
]
