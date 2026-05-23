import matplotlib.pyplot as plt
import numpy as np
from src.lle import lle


def lle_embedding():
    datasets = [
        {
            'name': 'fishbowl',
            'file': 'fishbowl_dense.npz',
            'dim': 3,
            'm': 2,
            'params': {'n_rule': 'knn', 'k': 20, 'tol': 1e-3}
        },
        {
            'name': 'swissroll',
            'file': 'swissroll_data.npz',
            'dim': 3,
            'm': 2,
            'params': {'n_rule': 'knn', 'k': 12, 'tol': 1e-3}
        },
        {
            'name': 'flatroll',
            'file': 'flatroll_data.npz',
            'dim': 2,
            'm': 1,
            'params': {'n_rule': 'knn', 'k': 10, 'tol': 1e-3}
        }
    ]

    for ds in datasets:
        try:
            data_file = np.load(ds['file'])
            
            if ds['name'] == 'fishbowl':
                X = data_file['X'].T
                ref = X[:, 2]
            elif ds['name'] == 'swissroll':
                X = data_file['x_noisefree'].T
                ref = data_file['z'].T[:, 0]
            elif ds['name'] == 'flatroll':
                X = data_file['Xflat'].T
                ref = data_file['true_embedding']

            Y = lle(X, ds['m'], ds['params']['tol'], ds['params']['n_rule'], k=ds['params'].get('k'))

            fig = plt.figure(figsize=(12, 5))
            plt.suptitle(f"Assignment 7: {ds['name'].capitalize()} LLE")

            if ds['dim'] == 3:
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=ref, cmap='viridis')
                ax1.set_title("Original 3D Data")
              
                ax2 = fig.add_subplot(122)
                ax2.scatter(Y[:, 0], Y[:, 1], c=ref, cmap='viridis')
                ax2.set_title("2D LLE Embedding")
            
            else:
                ax1 = fig.add_subplot(121)
                ax1.scatter(X[:, 0], X[:, 1], c=ref, cmap='viridis')
                ax1.set_title("Original 2D Data")
                
                ax2 = fig.add_subplot(122)
                ax2.scatter(ref, Y[:, 0], c=ref, cmap='viridis', alpha=0.6)
                ax2.set_xlabel("Ground Truth Reference")
                ax2.set_ylabel("1D LLE Embedding")
                ax2.set_title("Reference vs 1D Embedding")

            plt.tight_layout()
            plt.show()

        except FileNotFoundError:
            print(f"File {ds['file']} not found. Skipping {ds['name']}.")
