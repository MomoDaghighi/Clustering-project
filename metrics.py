import numpy as np


def clustering_score(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    if true_labels.size != predicted_labels.size:
        raise ValueError("The number of true and predicted labels must be the same.")

    # Get unique labels
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)

    # This will store the match between predicted and true labels
    match = {}

    # Iterate over each unique label in true labels
    for true_label in unique_true:
        # This will store the score for each potential match
        scores = {}

        # Iterate over each unique label in predicted labels
        for pred_label in unique_pred:
            if pred_label not in match:
                # Count the number of times the current predicted label matches the current true label
                score = np.sum((predicted_labels == pred_label) & (true_labels == true_label))
                scores[pred_label] = score

        # Find the predicted label that has the most matches with the current true label
        if scores:
            best_pred_label = max(scores, key=scores.get)
            match[best_pred_label] = true_label

    # Now, create a new array with the predicted labels mapped to the true labels
    mapped_predictions = np.zeros_like(predicted_labels)
    for pred_label, true_label in match.items():
        mapped_predictions[predicted_labels == pred_label] = true_label

    # Calculate the clustering score
    score = np.sum(mapped_predictions == true_labels) / len(true_labels)

    return score
