import numpy as np

class PCA():
    def __init__(self, Xtrain):
        self.C = np.mean(Xtrain, axis=0) 
        X_centered = Xtrain - self.C
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        self.D = eigenvalues[idx]
        self.U = eigenvectors[:, idx]

    def project(self, Xtest, m):
        X_centered = Xtest - self.C
        U_m = self.U[:, :m]        
        Z = np.dot(X_centered, U_m)
        return Z

    def denoise(self, Xtest, m):
        Z = self.project(Xtest, m)
        U_m = self.U[:, :m]
        Y = np.dot(Z, U_m.T) + self.C
        return Y