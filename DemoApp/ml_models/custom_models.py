import numpy as np
from collections import Counter
class DecisionNode:
    """Class to represent a single node in a decision tree."""
    def __init__(self, left, right, feature_index, threshold, class_label=None):
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.class_label = class_label

    def decide(self, feature):
        if self.class_label is not None:
            return self.class_label
        # Add checks for None left/right nodes before calling decide
        if feature[self.feature_index] < self.threshold:
            if self.left:
                return self.left.decide(feature)
            else:
                 # If left is None, return the majority class of the current node's data
                 # (This is a fallback, ideally the tree building prevents this)
                 return None # Or handle appropriately, maybe raise an error or return default class
        else:
            if self.right:
                return self.right.decide(feature)
            else:
                 # If right is None, return the majority class of the current node's data
                 return None # Or handle appropriately

def gini_impurity(class_vector):
    if len(class_vector) == 0:
        return 0
    counts = Counter(class_vector)
    # Ensure counts are handled for both 0 and 1 classes
    prob_zero = counts.get(0, 0) / len(class_vector)
    prob_one = counts.get(1, 0) / len(class_vector)
    prob_sum = prob_zero ** 2 + prob_one ** 2
    return 1 - prob_sum

def partition_classes(features, classes, feature_index, threshold):
    # Ensure features and classes are numpy arrays if they are not already
    features_np = np.array(features)
    classes_np = np.array(classes)

    left_indices = features_np[:, feature_index] < threshold
    right_indices = ~left_indices

    # Ensure you handle the case where one partition might be empty
    X_left = features_np[left_indices]
    X_right = features_np[right_indices]
    y_left = classes_np[left_indices]
    y_right = classes_np[right_indices]

    return X_left, X_right, y_left, y_right

def gini_gain(previous_classes, current_classes_list):
    if len(previous_classes) == 0:
        return 0
    previous_gini = gini_impurity(previous_classes)
    current_gini = 0
    total_samples = len(previous_classes)
    for cls in current_classes_list:
        if len(cls) > 0:
            current_gini += (len(cls) / total_samples) * gini_impurity(cls)
    return previous_gini - current_gini

def get_most_occurring_feature(classes):
    if len(classes) == 0:
        return 0 # Or None, depending on how you handle empty nodes
    classes_np = classes if isinstance(classes, np.ndarray) else np.array(classes)
    # Handle the case where Counter might be empty if classes are empty (should be caught by len(classes)==0 check)
    counts = Counter(classes_np)
    if not counts:
        return 0 # Default or handle as needed
    return counts.most_common(1)[0][0]

class DecisionTree:
    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        # Ensure features and classes are numpy arrays at the start of fit
        features_np = np.array(features)
        classes_np = np.array(classes)
        self.root = self.__build_tree__(features_np, classes_np)

    def __build_tree__(self, features, classes, depth=0):
        # Base cases for recursion
        if len(classes) == 0: # No data left
            return DecisionNode(None, None, None, None, None) # Or a default class like 0
        if depth >= self.depth_limit: # Reached depth limit
            return DecisionNode(None, None, None, None, get_most_occurring_feature(classes))
        if np.all(classes == classes[0]): # All classes are the same
            return DecisionNode(None, None, None, None, classes[0])

        best_gain = -1
        best_feature = -1
        best_threshold = -1

        # Iterate through features and find the best split
        for feature_idx in range(features.shape[1]):
            thresholds = np.unique(features[:, feature_idx])
            for threshold in thresholds:
                # Call partition_classes with all required arguments
                _, _, y_left, y_right = partition_classes(features, classes, feature_idx, threshold)
                gain = gini_gain(classes, [y_left, y_right])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no split improves the gain, make a leaf node
        if best_gain == 0:
            return DecisionNode(None, None, None, None, get_most_occurring_feature(classes))

        # Recursively build left and right subtrees
        X_left, X_right, y_left, y_right = partition_classes(features, classes, best_feature, best_threshold)
        # Pass the next depth level to the recursive calls
        left_tree = self.__build_tree__(X_left, y_left, depth + 1)
        right_tree = self.__build_tree__(X_right, y_right, depth + 1)

        # Return the current node
        return DecisionNode(left_tree, right_tree, best_feature, best_threshold)


    def predict(self, features):
        features_np = features if isinstance(features, np.ndarray) else np.array(features)
        # Handle potential None results from decide method in leaf nodes
        results = [self.root.decide(feature) for feature in features_np]
        # Convert None results if necessary (e.g., to a default class)
        # For now, let's assume decide always returns a class label or None
        # You might want to filter out Nones or handle them based on your logic
        return [r for r in results if r is not None]


class ANN:
    def __init__(self, input_size, hidden1_size=32, hidden2_size=16, output_size=1, dropout_rate=0.2, learning_rate=0.01):
        # Initialize weights with He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss_history = []  # Store loss per epoch
        self.accuracy_history = []  # Store accuracy per epoch

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        x = np.array(x, dtype=np.float64)
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def dropout(self, x, training=True):
        if training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x

    def forward(self, X, training=True):
        X = np.array(X, dtype=np.float64)
        # Hidden layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.a1_drop = self.dropout(self.a1, training)

        # Hidden layer 2
        self.z2 = np.dot(self.a1_drop, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.a2_drop = self.dropout(self.a2, training)

        # Output layer
        self.z3 = np.dot(self.a2_drop, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    def backward(self, X, y, output):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        m = X.shape[0]

        # Reshape y to be a column vector (batch_size, 1)
        y = y.reshape(-1, 1) # Add this line

        # Gradient for output layer
        self.error = output - y
        self.delta3 = self.error * self.sigmoid_derivative(output)

        # Gradient for hidden layer 2
        self.delta2 = np.dot(self.delta3, self.W3.T) * self.relu_derivative(self.a2)
        self.delta2 *= (self.a2_drop > 0) # Apply dropout mask derivative

        # Gradient for hidden layer 1
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.relu_derivative(self.a1)
        self.delta1 *= (self.a1_drop > 0) # Apply dropout mask derivative

        # Update weights and biases
        self.W3 -= self.learning_rate * np.dot(self.a2_drop.T, self.delta3) / m
        self.b3 -= self.learning_rate * np.sum(self.delta3, axis=0, keepdims=True) / m
        self.W2 -= self.learning_rate * np.dot(self.a1_drop.T, self.delta2) / m
        self.b2 -= self.learning_rate * np.sum(self.delta2, axis=0, keepdims=True) / m
        self.W1 -= self.learning_rate * np.dot(X.T, self.delta1) / m
        self.b1 -= self.learning_rate * np.sum(self.delta1, axis=0, keepdims=True) / m

    def train(self, X, y, epochs=100, batch_size=32):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        m = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Train in batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward
                output = self.forward(X_batch, training=True)

                # Backward
                self.backward(X_batch, y_batch, output)

            # Calculate loss and accuracy on full training set (no dropout for evaluation)
            output = self.forward(X, training=False)
            # Add a small epsilon to prevent log(0)
            loss = -np.mean(y * np.log(output + 1e-10) + (1 - y) * np.log(1 - output + 1e-10))
            predictions = (output >= 0.5).astype(int).flatten() # Flatten predictions
            accuracy = np.mean(predictions == y)

            # Store metrics
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)


            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        return self.forward(X, training=False)

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        output = self.forward(X, training=False)
        return (output >= 0.5).astype(int).flatten() # Flatten output


def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, degree=3):
    return (1 + np.dot(x, y)) ** degree

def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

class KernelSVM:
    def __init__(self, kernel=linear_kernel, C=1.0, max_iters=1000, lr=0.001, **kernel_params):
        self.kernel = kernel
        self.C = C
        self.max_iters = max_iters
        self.lr = lr
        self.kernel_params = kernel_params
        self.sv_alpha = None 
        self.sv_X = None
        self.sv_y = None
        self.alpha = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)

        # Precompute Kernel Matrix
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i, j] = self.kernel(X[i], X[j], **self.kernel_params)

        # Training via SGD on dual hinge loss
        for _ in range(self.max_iters):
            for i in range(n_samples):
                margin = y[i] * np.sum(self.alpha * y * self.K[:, i])
                if margin < 1:
                    self.alpha[i] += self.lr * (1 - margin)
                else:
                    self.alpha[i] -= self.lr * self.C * self.alpha[i]  # regularization

        # Support vectors
        self.support_indices = self.alpha > 1e-5
        self.sv_alpha = self.alpha[self.support_indices]
        self.sv_X = X[self.support_indices]
        self.sv_y = y[self.support_indices]

    def project(self, X):
        result = []
        for x in X:
            s = 0
            for alpha_i, y_i, x_i in zip(self.sv_alpha, self.sv_y, self.sv_X):
                s += alpha_i * y_i * self.kernel(x, x_i, **self.kernel_params)
            result.append(s)
        return np.array(result)

    def predict(self, X):
        return np.sign(self.project(X))

# Định nghĩa các hàm khoảng cách
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Lớp KNNScratch đã được sửa
class KNNScratch:
    def __init__(self, k=3, weights='distance', metric='euclidean'):
        self.k = k
        self.weights = weights  # 'uniform' hoặc 'distance'
        self.metric = metric  # 'euclidean' hoặc 'manhattan'

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Chọn hàm khoảng cách dựa trên metric
        if self.metric == 'euclidean':
            distance_func = euclidean_distance
        elif self.metric == 'manhattan':
            distance_func = manhattan_distance
        else:
            raise ValueError("Metric must be 'euclidean' or 'manhattan'")

        # Tính khoảng cách giữa x và tất cả các điểm trong tập huấn luyện
        distances = [distance_func(x, x_train) for x_train in self.X_train]
        # Lấy k chỉ số của các điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]

        # Lấy nhãn và khoảng cách của các điểm gần nhất
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        # Áp dụng trọng số
        if self.weights == 'distance':
            # Tránh chia cho 0 bằng cách thêm một giá trị nhỏ (1e-5)
            weights = [1 / (d + 1e-5) if d > 0 else 1.0 for d in k_nearest_distances]
        else:  # 'uniform'
            weights = [1.0] * self.k

        # Tính nhãn phổ biến nhất với trọng số
        weighted_votes = Counter({label: w for label, w in zip(k_nearest_labels, weights)})
        most_common = weighted_votes.most_common(1)
        return most_common[0][0]
    

# Kernel functions
def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, degree=3):
    return (1 + np.dot(x, y)) ** degree

def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

class KernelSVM:
    def __init__(self, kernel=linear_kernel, C=1.0, max_iters=1000, lr=0.001, **kernel_params):
        self.kernel = kernel
        self.C = C
        self.max_iters = max_iters
        self.lr = lr
        self.kernel_params = kernel_params
        self.sv_alpha = None 
        self.sv_X = None
        self.sv_y = None
        self.alpha = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)

        # Precompute Kernel Matrix
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i, j] = self.kernel(X[i], X[j], **self.kernel_params)

        # Training via SGD on dual hinge loss
        for _ in range(self.max_iters):
            for i in range(n_samples):
                margin = y[i] * np.sum(self.alpha * y * self.K[:, i])
                if margin < 1:
                    self.alpha[i] += self.lr * (1 - margin)
                else:
                    self.alpha[i] -= self.lr * self.C * self.alpha[i]  # regularization

        # Support vectors
        self.support_indices = self.alpha > 1e-5
        self.sv_alpha = self.alpha[self.support_indices]
        self.sv_X = X[self.support_indices]
        self.sv_y = y[self.support_indices]

    def project(self, X):
        result = []
        for x in X:
            s = 0
            for alpha_i, y_i, x_i in zip(self.sv_alpha, self.sv_y, self.sv_X):
                s += alpha_i * y_i * self.kernel(x, x_i, **self.kernel_params)
            result.append(s)
        return np.array(result)

    def predict(self, X):
        return np.sign(self.project(X))