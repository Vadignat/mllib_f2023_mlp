def accuracy(predictions, labels):
    """
        Вычисление точности:
            accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
        TODO: реализуйте подсчет accuracy
    """

    correct_predictions = (predictions == labels).sum().item()
    total_samples = len(labels)
    accuracy_value = correct_predictions / total_samples
    return accuracy_value

