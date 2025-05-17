import numpy as np

def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis=2)**2)

def svm_optimization(X, y, kernel, C=1.0, tol=1e-3, max_passes=5):
    m, n = X.shape
    alphas = np.zeros(m)
    b = 0
    passes = 0
    K = kernel(X, X)

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            Ei = np.sum(alphas * y * K[:, i]) + b - y[i]
            if (y[i]*Ei < -tol and alphas[i] < C) or (y[i]*Ei > tol and alphas[i] > 0):
                j = np.random.choice([k for k in range(m) if k != i])
                Ej = np.sum(alphas * y * K[:, j]) + b - y[j]
                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    continue

                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = np.clip(alphas[j], L, H)
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])
                b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * K[i,i] - y[j] * (alphas[j] - alpha_j_old) * K[i,j]
                b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * K[i,j] - y[j] * (alphas[j] - alpha_j_old) * K[j,j]
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    return alphas, b

class CustomSVM:
    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        self.alphas = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.kernel = lambda x1, x2: rbf_kernel(x1, x2, gamma=self.gamma)

    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)  # Convert labels from 0/1 to -1/+1
        self.X = X
        self.y = y
        self.alphas, self.b = svm_optimization(X, y, self.kernel, self.C)
        support_idx = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[support_idx]
        self.support_vector_labels = y[support_idx]
        self.alphas = self.alphas[support_idx]

    def predict(self, X):
        K = self.kernel(self.support_vectors, X)
        y_pred = np.sum(self.alphas[:, None] * self.support_vector_labels[:, None] * K, axis=0) + self.b
        return (np.sign(y_pred) + 1) // 2  # Convert -1/+1 to 0/1

# X, y là dữ liệu huấn luyện, y ∈ {0, 1}
svm_model = CustomSVM(C=1.0, gamma=0.5)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)
