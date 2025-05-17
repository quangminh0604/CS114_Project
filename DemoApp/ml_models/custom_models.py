import numpy as np
from collections import Counter
import random
from sklearn.base import BaseEstimator, ClassifierMixin

# Logistic Regression from scratch
class LogisticRegressionScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Khởi tạo tham số
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Gradient
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Cập nhật tham số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(linear_model)
        return np.vstack([1-probs, probs]).T

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)[:, 1]
        return np.where(y_pred_prob >= 0.5, 1, 0)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Decision Node for Decision Tree
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

# Decision Tree from scratch
class DecisionTreeScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2, n_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  
        self.root = None
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
        return self

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
    
    def predict_proba(self, X):
        predictions = self.predict(X)
        probas = np.zeros((len(X), 2))
        probas[np.arange(len(X)), predictions] = 1
        return probas

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Random Forest from scratch
class RandomForestScratch(BaseEstimator, ClassifierMixin):
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

        for i in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)  # Bootstrap
            X_sample = X[idxs]
            y_sample = y[idxs]
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
                random_state=self.random_state + i if self.random_state else None
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)
    
    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Convert to probability based on proportion of votes
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        probs = np.zeros((len(X), 2))
        
        for i, tree_pred in enumerate(tree_preds):
            counts = Counter(tree_pred)
            probs[i, 0] = counts.get(0, 0) / len(self.trees)
            probs[i, 1] = counts.get(1, 0) / len(self.trees)
            
        return probs

# SVM from scratch
def rbf_kernel(x1, x2, gamma=1.0):
    # Reshape for broadcasting
    x1_reshaped = x1.reshape(x1.shape[0], 1, x1.shape[1])
    x2_reshaped = x2.reshape(1, x2.shape[0], x2.shape[1])
    # Calculate squared Euclidean distance
    squared_diff = np.sum((x1_reshaped - x2_reshaped) ** 2, axis=2)
    # Apply RBF kernel formula
    return np.exp(-gamma * squared_diff)

class SVMScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma=1.0, tol=1e-3, max_passes=5):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        # Convert labels to -1, 1
        self.classes_ = np.unique(y)
        y_transformed = np.where(y == 0, -1, 1)
        
        # Compute the kernel matrix
        K = rbf_kernel(X, X, self.gamma)
        
        m = X.shape[0]
        self.alphas = np.zeros(m)
        self.b = 0
        passes = 0
        
        # SMO Algorithm
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                # Calculate error
                E_i = np.sum(self.alphas * y_transformed * K[i, :]) + self.b - y_transformed[i]
                
                # Check if we can optimize alpha[i]
                if (y_transformed[i] * E_i < -self.tol and self.alphas[i] < self.C) or \
                   (y_transformed[i] * E_i > self.tol and self.alphas[i] > 0):
                    
                    # Select j != i randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, m)
                    
                    # Calculate error for j
                    E_j = np.sum(self.alphas * y_transformed * K[j, :]) + self.b - y_transformed[j]
                    
                    # Save old alphas
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    
                    # Compute L and H
                    if y_transformed[i] != y_transformed[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha j
                    self.alphas[j] -= y_transformed[j] * (E_i - E_j) / eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha i
                    self.alphas[i] += y_transformed[i] * y_transformed[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update threshold b
                    b1 = self.b - E_i - y_transformed[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                         y_transformed[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y_transformed[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                         y_transformed[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        # Get support vectors
        sv_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y_transformed[sv_indices]
        self.alphas = self.alphas[sv_indices]
        self.X_train = X
        
        return self

    def predict(self, X):
        # Use the kernel to compute predictions
        y_pred = np.zeros(X.shape[0])
        
        # Get support vector indices
        sv_indices = np.where(self.alphas > 1e-5)[0]
        
        if len(sv_indices) > 0:
            # Compute kernel between support vectors and test points
            K = rbf_kernel(self.support_vectors, X, self.gamma)
            
            # Calculate prediction
            for i in range(X.shape[0]):
                prediction = np.sum(self.alphas * self.support_vector_labels * K[:, i]) + self.b
                y_pred[i] = 1 if prediction > 0 else -1
        
        # Convert back to original labels
        return np.where(y_pred == -1, 0, 1)
    
    def predict_proba(self, X):
        # This is a crude approximation since SVM doesn't naturally output probabilities
        raw_scores = self._decision_function(X)
        # Apply sigmoid function to get probabilities
        probs_positive = 1 / (1 + np.exp(-raw_scores))
        return np.vstack([1 - probs_positive, probs_positive]).T
    
    def _decision_function(self, X):
        # Get support vector indices
        sv_indices = np.where(self.alphas > 1e-5)[0]
        
        if len(sv_indices) > 0:
            # Compute kernel between support vectors and test points
            K = rbf_kernel(self.support_vectors, X, self.gamma)
            
            # Calculate decision function
            decision_values = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                decision_values[i] = np.sum(self.alphas * self.support_vector_labels * K[:, i]) + self.b
            
            return decision_values
        else:
            return np.zeros(X.shape[0])