import numpy as np

class PCA:
    def __init__(self, k=None):
        self.k = k
        self.mean = None
        self.std = None
        self.components = None
        self.S = None

    def fit(self, X):
        X = X / 255

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)

        self.std[self.std == 0] = 1.0

        X_norm = (X - self.mean) / self.std

        U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
        self.S = S

        if self.k is not None:
            self.components = Vt[:self.k]
        else:
            self.components = Vt

        return np.dot(X_norm, self.components.T)

    def transform(self, X_new):
        X_new = X_new / 255
        X_norm = (X_new - self.mean) / self.std
        return np.dot(X_norm, self.components.T)

    def inverse_transform(self, Z):
        X_norm_recon = np.dot(Z, self.components)
        return (X_norm_recon * self.std + self.mean) * 255
    
class XGBoostTree:
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
        
        def is_leaf_node(self):
            return self.value is not None
        
    def __init__(self, max_depth: int=3, min_samples_split: int=2, gamma: float=0.0, reg_lambda: float=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.gamma = gamma
        self.reg_lambda = reg_lambda
    
    def fit(self, X, g, h):
        self.n, self.m = X.shape
        self.root = self._build_tree(X, g, h, 0)

    def _build_tree(self, X, g, h, depth):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            leaf_value = self._compute_leaf_value(g, h)
            return self.Node(value=leaf_value)
        
        best_feat, best_thresh = self._best_split(X, g, h)
        if best_feat is None or best_thresh is None:
            leaf_value = self._compute_leaf_value(g, h)
            return self.Node(value=leaf_value)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        child = self.Node(feature_index=best_feat, threshold=best_thresh,
                          left=self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth+1),
                          right=self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth+1))
        return child

    def _best_split(self, X, g, h):
        m, n = X.shape
        best_gain = -np.inf
        best_thresh, best_feat = None, None

        feature_indices = list(range(n))

        for feat_idx in feature_indices: # Iterating over features

            X_col = X[:, feat_idx]
            sorted_idx = np.argsort(X[:, feat_idx])

            X_sorted, g_sorted, h_sorted = X_col[sorted_idx], g[sorted_idx], h[sorted_idx]

            G_prefix = np.cumsum(g_sorted)
            H_prefix = np.cumsum(h_sorted)
            G_total = G_prefix[-1]
            H_total = H_prefix[-1]

            for i in range(1, len(X_sorted)):
                if X_sorted[i] == X_sorted[i-1]:
                    continue

                GL, HL = G_prefix[i-1], H_prefix[i-1]
                GR, HR = G_total - GL, H_total - HL

                gain = 0.5 * (
                    (GL ** 2 / (HL + self.reg_lambda + 1e-8)) +
                    (GR ** 2 / (HR + self.reg_lambda + 1e-8)) -
                    (G_total ** 2 / (H_total + self.reg_lambda + 1e-8))
                ) - self.gamma

                if gain > best_gain and gain >= 0:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = (X_sorted[i] + X_sorted[i-1]) / 2

        return best_feat, best_thresh
    
    def _compute_leaf_value(self, g, h):
        G = np.sum(g)
        H = np.sum(h)
        return -G / (H + self.reg_lambda)
        
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        preds = np.zeros(X.shape[0])
        for i in range(len(X)):
            node = self.root
            while not node.is_leaf_node():
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds[i] = node.value
        return preds

class XGBoostClassifier:
    def __init__(
            self,
            n_estimators=10,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split: int=2,
            gamma: float=0.0,
            reg_lambda: float=1.0,
            threshold: float=0.5,
            colsample=1.0,
            random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.threshold = threshold
        self.min_samples_split = min_samples_split
        self.sigmoid = lambda z: 1 / (1 + np.exp(-z))
        self.colsample = colsample

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        eps = 1e-15
        y_mean = np.clip(np.mean(y), eps, 1 - eps)
        self.init_pred = np.log(y_mean / (1 - y_mean))
        pred = np.full_like(y, fill_value=self.init_pred, dtype=float)

        for _ in range(self.n_estimators):
            prob = self.sigmoid(pred)
            g = prob - y
            h = prob * (1 - prob)
            
            n_features = int(self.colsample *  X.shape[1])
            col_idx = np.random.choice(X.shape[1], n_features, replace=False)
            X_sub = X[:, col_idx]
            tree = XGBoostTree(
                self.max_depth,
                self.min_samples_split,
                self.gamma,
                self.reg_lambda
            )
            tree.fit(X_sub, g, h)

            pred += self.learning_rate * tree.predict(X[:, col_idx])
            self.trees.append((tree, col_idx))
        

    def predict(self, X):
        X = np.array(X)
        prob = self._predict_proba(X)
        result = np.where(prob <= self.threshold, 0, 1)
        return result
    
    def _predict_proba(self, X):
        pred = np.full((X.shape[0],), self.init_pred, dtype=float)
        for tree, cols in self.trees:
            pred += self.learning_rate * tree.predict(X[:, cols])
        prob = self.sigmoid(pred)
        return prob

class SoftmaxRegression:
    def __init__(self, lr=0.1, num_iters=1000, reg=0.0):
        self.lr = lr
        self.num_iters = num_iters
        self.reg = reg
        self.W = None
        self.b = None

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        self.W = np.zeros((n_features, num_classes))
        self.b = np.zeros((1, num_classes))

        y_onehot = np.eye(num_classes)[y]

        for i in range(self.num_iters):

            scores = np.dot(X, self.W) + self.b
            probs = self._softmax(scores)

            grad_scores = (probs - y_onehot) / n_samples

            dW = np.dot(X.T, grad_scores) + self.reg * self.W
            db = np.sum(grad_scores, axis=0, keepdims=True)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if i % 200 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

    def predict(self, X):
        X = np.array(X)
        probs = self._predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def _predict_proba(self, X):
        scores = np.dot(X, self.W) + self.b
        probs = self._softmax(scores)
        return probs
