# models.py

from sentiment_data import *
from utils import *

from collections import Counter
from typing import List
import numpy as np
import random

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract unigram features from a sentence.
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should add new features to the indexer
        :return: A Counter mapping feature indices to their counts
        """
        feature_counter = Counter()
        for word in sentence:
            # Add word to indexer if needed, get its index
            word_idx = self.indexer.add_and_get_index(word, add=add_to_indexer)
            # Only count the word if it was successfully indexed
            if word_idx >= 0:
                feature_counter[word_idx] += 1
        return feature_counter


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feature_counter = Counter()
        for i in range(len(sentence) - 1):
            bigram = f"{sentence[i]}_{sentence[i + 1]}"
            bigram_idx = self.indexer.add_and_get_index(bigram, add=add_to_indexer)
            if bigram_idx >= 0:
                feature_counter[bigram_idx] += 1
        return feature_counter


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        #lowercasing
        #drop very short tokens and purely numeric tokens
        #count-based features
        feature_counter = Counter()
        for word in sentence:
            w = word.lower()
            if len(w) <= 2 or w.isdigit():
                continue
            idx = self.indexer.add_and_get_index(w, add=add_to_indexer)
            if idx >= 0:
                feature_counter[idx] += 1
        return feature_counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Perceptron classifier that maintains weights learned during training.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        """
        :param weights: numpy array of weight values
        :param feat_extractor: FeatureExtractor to use for extracting features at test time
        """
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        """
        Predict the sentiment of a sentence using the learned weights.
        :param sentence: words in the sentence to classify
        :return: 1 for positive, 0 for negative
        """
        # Extract features from the sentence (don't add new features at test time)
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        
        # Compute dot product of features and weights
        score = 0.0
        for feature_idx, feature_count in features.items():
            score += self.weights[feature_idx] * feature_count
        
        # Return 1 if score is positive, 0 otherwise
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        """
        :param weights: numpy array of weight values
        :param feat_extractor: FeatureExtractor to use for extracting features at test time
        """
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        """
        Predict the sentiment of a sentence using the learned weights.
        :param sentence: words in the sentence to classify
        :return: 1 for positive, 0 for negative
        """
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0.0
        for feature_idx, feature_count in features.items():
            score += self.weights[feature_idx] * feature_count
        prob = 1.0 / (1.0 + np.exp(-score))
        return 1 if prob >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron algorithm.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # First pass: extract features from all training examples to build the feature indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    # Initialize weight vector with zeros using numpy array
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)
    
    # Learning rate schedule parameters
    initial_lr = 1.0
    lr_decay_factor = 0.8
    lr_decay_every = 1  # Decay learning rate every epoch for more gradual decrease
    
    # Perceptron training: multiple epochs
    num_epochs = 5
    for epoch in range(num_epochs):
        # Update learning rate based on schedule
        learning_rate = initial_lr * (lr_decay_factor ** (epoch // lr_decay_every))
        
        # Randomly shuffle training data each epoch
        shuffled_exs = train_exs.copy()
        random.shuffle(shuffled_exs)
        
        for ex in shuffled_exs:
            # Extract features for this example
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            
            # Compute prediction: score = w · x
            score = 0.0
            for feature_idx, feature_count in features.items():
                score += weights[feature_idx] * feature_count
            
            # Perceptron prediction: 1 if score > 0, else 0
            prediction = 1 if score > 0 else 0
            
            # Update weights if prediction is wrong
            # Convert label to {-1, 1} for the update rule: w = w + lr * y * x
            if prediction != ex.label:
                label_sign = 2 * ex.label - 1  # Convert 0/1 to -1/1
                for feature_idx, feature_count in features.items():
                    weights[feature_idx] += learning_rate * label_sign * feature_count
    
    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Set random seed for reproducibility
    random.seed(42)

    # First pass: extract features from all training examples to build the feature indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Initialize weight vector with zeros using numpy array
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    # Training parameters
    initial_lr = 0.15
    lr_decay_factor = 0.9
    lr_decay_every = 1  
    num_epochs = 15

    for epoch in range(num_epochs):
        learning_rate = initial_lr * (lr_decay_factor ** (epoch // lr_decay_every))
        shuffled_exs = train_exs.copy()
        random.shuffle(shuffled_exs)

        for ex in shuffled_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            score = 0.0
            for feature_idx, feature_count in features.items():
                score += weights[feature_idx] * feature_count

            prob = 1.0 / (1.0 + np.exp(-score))
            error = ex.label - prob

            for feature_idx, feature_count in features.items():
                weights[feature_idx] += learning_rate * error * feature_count

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model