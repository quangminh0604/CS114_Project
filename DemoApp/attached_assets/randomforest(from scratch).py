import numpy as np
from collections import Counter
import random

import numpy as np
from collections import Counter

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

def gini(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum(ps ** 2)

def split(X, y, feat_idx, threshold):
    left_idxs = X[:, feat_idx] <= threshold
    right_idxs = X[:, feat_idx] > threshold
    return X[left_idxs], y[left_idxs], X[right_idxs], y[right_idxs]

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  
        self.root = None
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else self.n_features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features_total = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        feat_idxs = self.rng.choice(n_features_total, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        X_left, y_left, X_right, y_right = split(X, y, best_feat, best_thresh)
        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)
        return DecisionNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for thr in thresholds:
                X_left, y_left, X_right, y_right = split(X, y, feat_idx, thr)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh

    def _information_gain(self, y, y_left, y_right):
        parent_loss = gini(y)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        child_loss = (n_l / n) * gini(y_left) + (n_r / n) * gini(y_right)
        return parent_loss - child_loss

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Rừng ngẫu nhiên
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, n_features=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.random_state = random_state

    def fit(self, X, y):
        self.trees = []
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)  # Bootstrap
            X_sample = X[idxs]
            y_sample = y[idxs]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)

rf_model = RandomForest(n_trees=100, max_depth=5, n_features=int(np.sqrt(X_train_scaled.shape[1])), random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)




