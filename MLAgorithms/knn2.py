# coding:utf-8

from collections import Counter
import numpy as np
import itertools
from scipy.spatial.distance import euclidean


class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()


class KNNBase(BaseEstimator):
    def __init__(self, k=5, max_k=9, min_k=3, distance_func=euclidean, weight_distance=True):
        """
        Base class for K Nearest Neighbors classifier with dynamic k
        """
        self.k = k
        self.max_k = max_k
        self.min_k = min_k
        self.distance_func = distance_func
        self.weight_distance = weight_distance
        self.upper_threshold = None
        self.lower_threshold = None

    def aggregate(self, neighbors_targets, neighbors_weights=None):
        raise NotImplementedError

    def _predict(self, X=None):
        predictions = [self._predict_x(x) for x in X]

        return np.array(predictions)

    def fit(self, X, y=None):
        super()._setup_input(X, y)
        # Calculate distance thresholds based on training data
        self._calculate_thresholds()

    def _calculate_thresholds(self):
        # Calculate distances between all pairs in training data
        distances = [self.distance_func(x, y) for x, y in itertools.combinations(self.X, 2)]
        distances = np.array(distances)
        self.upper_threshold = np.percentile(distances, 75)  # 75th percentile
        self.lower_threshold = np.percentile(distances, 25)  # 25th percentile

    def _predict_x(self, x):
        distances = np.array([self.distance_func(x, example) for example in self.X])
        sorted_neighbors = sorted(((dist, target) for (dist, target) in zip(distances, self.y)), key=lambda x: x[0])

        dynamic_k = self.k
        neighbors_targets = []
        neighbors_weights = []

        while True:
            avg_distance = np.mean([dist for (dist, _) in sorted_neighbors[:dynamic_k]])
            if avg_distance > self.upper_threshold and dynamic_k < self.max_k:
                dynamic_k = min(self.max_k, dynamic_k + 2)
            elif avg_distance < self.lower_threshold and dynamic_k > self.min_k:
                dynamic_k = max(self.min_k, dynamic_k - 2)
            else:
                break

        neighbors_targets = [target for (_, target) in sorted_neighbors[:dynamic_k]]
        neighbors_weights = [1/dist if dist != 0 else 1e-5 for (dist, _) in sorted_neighbors[:dynamic_k]]

        return self.aggregate(neighbors_targets, neighbors_weights if self.weight_distance else None)
        

class KNNVariant(KNNBase):
    """
    Nearest neighbors classifier.

    Note: if there is a tie for the most common label among the neighbors, then
    the predicted label is arbitrary. This class extends KNNBase and implements
    the aggregate method for making predictions based on neighbor voting.
    """

    def __init__(self, k=5, max_k=10, min_k=3, distance_func=euclidean, weight_distance=True):
        """
        Initialize the KNNVariant with the same parameters as KNNBase,
        ensuring all are passed correctly.
        """
        super().__init__(k=k, max_k=max_k, min_k=min_k, distance_func=distance_func, weight_distance=weight_distance)

    def aggregate(self, neighbors_targets, neighbors_weights=None):
        """
        Return the most common target label, considering weights if provided.
        """
        if neighbors_weights:
            weighted_vote = Counter()
            for label, weight in zip(neighbors_targets, neighbors_weights):
                weighted_vote[label] += weight
            most_common_label = weighted_vote.most_common(1)[0][0]
        else:
            most_common_label = Counter(neighbors_targets).most_common(1)[0][0]
        return most_common_label
