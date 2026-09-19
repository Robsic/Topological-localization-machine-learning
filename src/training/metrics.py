from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
)


def compute_classification_metrics(
    expected: Sequence[int],
    predicted: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> dict[str, object]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        raise ValueError("at least one prediction is required")
    if labels is None:
        labels = sorted(set(expected).union(predicted))
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(
            recall_score(
                expected,
                predicted,
                labels=sorted(set(expected)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "per_class": classification_report(
            expected, predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }


def compute_pipeline_metrics(
    expected_stage1: Sequence[int],
    predicted_stage1: Sequence[int],
    expected_nodes: Sequence[int | None],
    predicted_nodes: Sequence[int | None],
) -> dict[str, object]:
    lengths = {len(expected_stage1), len(predicted_stage1), len(expected_nodes), len(predicted_nodes)}
    if len(lengths) != 1:
        raise ValueError("all pipeline inputs must have the same length")
    if not expected_stage1:
        raise ValueError("at least one pipeline prediction is required")
    stage1 = compute_classification_metrics(expected_stage1, predicted_stage1, labels=[0, 1])

    intersection_positions = [index for index, value in enumerate(expected_stage1) if value == 1]
    node_expected = [expected_nodes[index] for index in intersection_positions]
    node_predicted = [predicted_nodes[index] for index in intersection_positions]
    valid_positions = [
        index
        for index, (expected, predicted) in enumerate(zip(node_expected, node_predicted))
        if expected is not None and predicted is not None
    ]
    stage2 = None
    if valid_positions:
        stage2 = compute_classification_metrics(
            [int(node_expected[index]) for index in valid_positions],
            [int(node_predicted[index]) for index in valid_positions],
        )

    end_to_end_correct = 0
    for true_class, predicted_class, true_node, predicted_node in zip(
        expected_stage1, predicted_stage1, expected_nodes, predicted_nodes
    ):
        if true_class == 0:
            end_to_end_correct += int(predicted_class == 0)
        else:
            end_to_end_correct += int(predicted_class == 1 and predicted_node == true_node)

    return {
        "stage1": stage1,
        "stage2_on_intersections": stage2,
        "end_to_end_accuracy": end_to_end_correct / len(expected_stage1),
        "number_of_samples": len(expected_stage1),
        "number_of_intersections": len(intersection_positions),
    }
