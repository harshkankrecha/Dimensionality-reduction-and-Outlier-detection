import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components

def lle(X, m, tol, n_rule, k=None, epsilon=None):
    n, d = X.shape
    dist_matrix = cdist(X, X, 'euclidean')
    adj_matrix = np.zeros((n, n), dtype=bool)
    if n_rule == 'knn':
        if k is None or k <= 0 or k >= n:
            raise ValueError("k must be a positive integer smaller than n.")
        
        for i in range(n):
            neighbors_idx = np.argsort(dist_matrix[i])[1:k+1]
            adj_matrix[i, neighbors_idx] = True
            
    elif n_rule == 'eps-ball':
        if epsilon is None or epsilon <= 0:
            raise ValueError("epsilon must be a positive float.")
        adj_matrix = (dist_matrix <= epsilon) & (dist_matrix > 0)
        
    else:
        raise ValueError("n_rule must be 'knn' or 'eps-ball'.")

    n_components, labels = connected_components(adj_matrix, directed=False)
    if n_components > 1:
        raise ValueError(f"The neighborhood graph is not connected ({n_components} components). "
                         "Increase k or epsilon.")

    
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.where(adj_matrix[i])[0]
        n_i = len(idx)
        if n_i == 0:
            continue
            
        
        Z = X[idx] - X[i]
        C = np.dot(Z, Z.T)
        
        
        trace = np.trace(C)
        r = tol * trace if trace > 0 else tol
        C += np.eye(n_i) * r
        
        w = np.linalg.solve(C, np.ones(n_i))
        w = w / np.sum(w)  
        W[i, idx] = w

    
    I = np.eye(n)
    M = np.dot((I - W).T, (I - W))
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    idx_sorted = np.argsort(eigenvalues)
    selected_indices = idx_sorted[1:m+1]
    Y = eigenvectors[:, selected_indices]
    return Y
