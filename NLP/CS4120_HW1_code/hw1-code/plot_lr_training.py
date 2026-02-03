import numpy as np
import matplotlib.pyplot as plt
import random

from sentiment_data import read_sentiment_examples
from models import UnigramFeatureExtractor
from utils import Indexer


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def compute_log_likelihood(weights, exs, feat_extractor):
    total = 0.0
    for ex in exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        score = 0.0
        for feature_idx, feature_count in features.items():
            score += weights[feature_idx] * feature_count
        prob = sigmoid(score)
        prob = min(max(prob, 1e-12), 1.0 - 1e-12)
        total += ex.label * np.log(prob) + (1 - ex.label) * np.log(1 - prob)
    return total


def compute_accuracy(weights, exs, feat_extractor):
    correct = 0
    for ex in exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        score = 0.0
        for feature_idx, feature_count in features.items():
            score += weights[feature_idx] * feature_count
        pred = 1 if sigmoid(score) >= 0.5 else 0
        if pred == ex.label:
            correct += 1
    return correct / len(exs)


def train_lr_with_tracking(train_exs, dev_exs, step_size, num_epochs=10):
    feat_extractor = UnigramFeatureExtractor(Indexer())
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    train_ll = []
    dev_acc = []

    for epoch in range(num_epochs):
        shuffled_exs = train_exs.copy()
        random.shuffle(shuffled_exs)

        for ex in shuffled_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = 0.0
            for feature_idx, feature_count in features.items():
                score += weights[feature_idx] * feature_count
            prob = sigmoid(score)
            error = ex.label - prob
            for feature_idx, feature_count in features.items():
                weights[feature_idx] += step_size * error * feature_count

        train_ll.append(compute_log_likelihood(weights, train_exs, feat_extractor))
        dev_acc.append(compute_accuracy(weights, dev_exs, feat_extractor))

    return train_ll, dev_acc


def main():
    random.seed(42)
    np.random.seed(42)

    train_exs = read_sentiment_examples("data/train.txt")
    dev_exs = read_sentiment_examples("data/dev.txt")

    step_sizes = [0.05, 0.1, 0.2]
    num_epochs = 10

    plt.figure(figsize=(12, 5))

    # Plot training log-likelihood
    plt.subplot(1, 2, 1)
    for lr in step_sizes:
        train_ll, dev_acc = train_lr_with_tracking(train_exs, dev_exs, lr, num_epochs)
        plt.plot(range(1, num_epochs + 1), train_ll, label=f"lr={lr}")
    plt.title("Training Log Likelihood")
    plt.xlabel("Epoch")
    plt.ylabel("Log Likelihood")
    plt.legend()

    # Plot dev accuracy
    plt.subplot(1, 2, 2)
    for lr in step_sizes:
        train_ll, dev_acc = train_lr_with_tracking(train_exs, dev_exs, lr, num_epochs)
        plt.plot(range(1, num_epochs + 1), dev_acc, label=f"lr={lr}")
    plt.title("Dev Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
