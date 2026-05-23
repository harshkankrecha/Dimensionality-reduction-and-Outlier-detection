import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from src.lle import lle


def plot_neighborhood_graph(ax, X, k, title):
    n = X.shape[0]
    dists = cdist(X, X)
    
    ax.scatter(X[:, 0], X[:, 1], s=5, c='black', alpha=0.3, zorder=5)
    
    print(f"  -> Drawing lines for: {title}")
    for i in range(n):
        if i % 100 == 0 or i == n-1:
            print(f"     {i+1}/{n} points processed", end='\r')
            
        neighbors_idx = np.argsort(dists[i])[1:k+1]
        
        for n_idx in neighbors_idx:
            ax.plot([X[i, 0], X[n_idx, 0]], [X[i, 1], X[n_idx, 1]], 
                    'r-', lw=0.5, alpha=0.2)        
    ax.set_title(title)

def lle_noise():
    try:
        data = np.load('flatroll_data.npz')
        X_orig = data['Xflat'].T
        ref = data['true_embedding']
    except FileNotFoundError:
        print("Error: 'flatroll_data.npz' not found. Please ensure it's in the current directory.")
        return

    variances = [0.2, 1.8]
    k_settings = [('Good k', 10), ('Too Large k', 50)]
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    row = 0

    for var in variances:
        print(f"\nAdding Gaussian noise with variance {var}")
        noise = np.random.normal(0, np.sqrt(var), X_orig.shape)
        X_noisy = X_orig + noise
        
        for k_label, k_val in k_settings:
            current_label = f"Var={var}, {k_label} (k={k_val})"
            plot_neighborhood_graph(axes[row, 0], X_noisy, k_val, 
                                   f"Neighborhood Graph\n({current_label})")
            try:
                Y = lle(X_noisy, m=1, tol=1e-3, n_rule='knn', k=k_val)
                
                axes[row, 1].scatter(ref, Y[:, 0], c=ref, cmap='viridis', s=10)
                axes[row, 1].set_xlabel("True Embedding (Reference)")
                axes[row, 1].set_ylabel("LLE 1D Result")
                axes[row, 1].set_title(f"LLE Embedding\n({current_label})")
            except ValueError as e:
                print(f"  !! LLE Error: {e}")
                axes[row, 1].text(0.5, 0.5, f"LLE Failed:\nGraph Disconnected", 
                                  ha='center', va='center', color='red')
                axes[row, 1].set_title(f"LLE Failed ({current_label})")
            
            row += 1

    print("\n" + "="*30)
    print("Rendering plots might take a moment...")
    print("="*30)
    
    plt.tight_layout()
    plt.show()
    