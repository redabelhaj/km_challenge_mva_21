import numpy as np
from scipy import optimize


def fit_binary(km, y, C=1e2):
    """
    Fit a binary SVM from kernel matrix km and labels y in {-1, 1}
    C : regularization coefficient
    Returns:
    - b (float) : intercept
    - beta (np.array) : coefficients
    """

    N = len(y)
    U = np.concatenate([np.eye(N), -np.eye(N)], axis=0)
    V = np.array(N * [0] + N * [C])

    def loss(alpha):
        return 0.5 * (alpha * y).T @ km @ (alpha * y) - alpha.sum()

    def grad_loss(alpha):
        return np.diag(y) @ km @ (y * alpha) - np.ones_like(alpha)

    fun_eq = lambda alpha: y.T @ alpha
    jac_eq = lambda alpha: y
    fun_ineq = lambda alpha: U @ alpha + V
    jac_ineq = lambda alpha: U

    constraints = (
        {"type": "eq", "fun": fun_eq, "jac": jac_eq},
        {"type": "ineq", "fun": fun_ineq, "jac": jac_ineq},
    )

    optRes = optimize.minimize(
        fun=lambda alpha: loss(alpha),
        x0=np.ones(N),
        method="SLSQP",
        jac=lambda alpha: grad_loss(alpha),
        constraints=constraints,
    )
    alpha = optRes.x

    supportIndices = np.argwhere((alpha > 1e-10) & (alpha < C)).squeeze()
    b = 0.0
    if supportIndices.size > 0:
        b = (y - km @ (alpha * y))[supportIndices][0]
    beta = alpha * y

    return b, beta


class MultiClassSVM:
    """
    Implements the Multi class SVM classifier with the OVO scheme
    """

    def __init__(self, kernel_matrix, n_classes=10, C=1e2):
        """
        kernel_matrix : K(X, X) matrix of the classifier
        """
        self.kxx = kernel_matrix
        self.n_classes = n_classes
        self.epsilon = 1e-3
        self.C = C

    def fit_global(self, y):
        """
        Computes the intercepts and coefficients for all binary SVM problems
        """
        self.y = y
        class_pairs = [
            {
                "class_pos": i,
                "class_neg": j,
            }
            for i in range(self.n_classes)
            for j in range(i)
        ]
        for pair in class_pairs:
            inds = np.argwhere(
                (self.y == pair["class_pos"]) | (self.y == pair["class_neg"])
            ).squeeze()
            ytr = self.y[inds]
            ytr = np.where(ytr == pair["class_pos"], 1, -1)
            kmtr = self.kxx[np.ix_(inds, inds)]
            b, beta = fit_binary(kmtr, ytr, C=self.C)
            pair["b"] = b
            pair["beta"] = beta
        self.class_pairs = class_pairs

    def predict_global(self, kernel_matrix):
        for pair in self.class_pairs:
            inds = np.argwhere(
                (self.y == pair["class_pos"]) | (self.y == pair["class_neg"])
            ).squeeze()
            kmtr = kernel_matrix[np.ix_(np.arange(kernel_matrix.shape[0]), inds)]
            pair["pred"] = np.where(
                (pair["b"] + kmtr @ pair["beta"]) > 0,
                pair["class_pos"],
                pair["class_neg"],
            )
        all_preds = np.concatenate(
            [x["pred"][None, :] for x in self.class_pairs], axis=0
        )
        return np.array(
            [np.bincount(all_preds[:, i]).argmax() for i in range(all_preds.shape[1])]
        )
