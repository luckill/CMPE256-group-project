import numpy as np

from utils import zscore_1d


class LinUCB:
    def __init__(self, d, alpha=0.01):
        self.d = d
        self.alpha = alpha
        self.A_inv = np.eye(d, dtype=np.float64)
        self.b = np.zeros(d, dtype=np.float64)

    def score(self, X):
        theta = self.A_inv @ self.b
        exploit = X @ theta
        explore = np.sqrt(np.einsum("ij,jk,ik->i", X, self.A_inv, X))
        return exploit + self.alpha * explore

    def update(self, X_selected, rewards_selected):
        for x, r in zip(X_selected, rewards_selected):
            x = np.asarray(x, dtype=np.float64)
            Ainv_x = self.A_inv @ x
            denom = 1.0 + (x @ Ainv_x)
            self.A_inv -= np.outer(Ainv_x, Ainv_x) / denom
            self.b += float(r) * x

def select_hybrid_top_k(event_df, X_event, bandit, top_k=10, lambda_ncf=0.85):
    bandit_scores = bandit.score(X_event)
    ncf_scores = event_df["ncf_score"].to_numpy(dtype=np.float64)

    ncf_z = zscore_1d(ncf_scores)
    bandit_z = zscore_1d(bandit_scores)

    final_scores = lambda_ncf * ncf_z + (1.0 - lambda_ncf) * bandit_z

    order = np.lexsort((
        event_df["rank"].to_numpy(dtype=np.int64),
        -ncf_scores,
        -final_scores,
    ))
    top_idx = order[: min(top_k, len(event_df))]

    return top_idx, bandit_scores, final_scores
