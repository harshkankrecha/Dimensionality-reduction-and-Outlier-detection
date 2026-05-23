from scipy.io import loadmat
from src.pca import PCA
import matplotlib.pyplot as plt
import numpy as np

def usps():
    usps_data = loadmat('usps.mat')
    print(usps_data.keys())
    X = usps_data['data_patterns']
    y = usps_data['data_labels']

    if X.shape[0]==256:
        X=X.T
    labels = y.flatten()

    pca = PCA(X)

    print(X.shape)
    print(labels.shape)

    plt.figure(figsize=(10,5))
    plt.plot(pca.D)

    plt.bar(range(25),pca.D[:25])

    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Largest principal values')

    plt.grid(True)
    plt.show()
    fig,axes = plt.subplots(1,5,figsize=(15,3))
    for i in range(5):
        eigendigit = pca.U[:,i].reshape(16,16)
        axes[i].imshow(eigendigit,cmap='gray')
        axes[i].set_title(f"PC {i+1}")
        axes[i].axis('off')

    plt.show()

    sigma_low = 0.3
    sigma_high = 1
    sigma_outlier = 5

    X_low_noise = X+sigma_low*np.random.randn(*X.shape)
    X_high_noise = X+sigma_high*np.random.randn(*X.shape)

    X_outliers = X.copy()

    indices = np.random.choice(len(X),5,replace=False)

    X_outliers[indices]+=sigma_outlier*np.random.randn(5,256)

    pca_low = PCA(X_low_noise)
    pca_high = PCA(X_high_noise)
    pca_outlier = PCA(X_outliers)

    plt.figure(figsize=(12,6))

    plt.plot(pca.D,label='Original')
    plt.plot(pca_low.D,label='Low noise')
    plt.plot(pca_high.D,label='High noise')
    plt.plot(pca_outlier.D,label='Outlier')

    plt.legend()

    plt.xlabel('Component')
    plt.ylabel('Eigenvalues')

    plt.title('Comparision of PCA spectra')

    plt.show()

    m = 30

    X_low_denoised = pca_low.denoise(X_low_noise, m)

    X_high_denoised = pca_high.denoise(X_high_noise, m)

    X_outlier_denoised = pca_outlier.denoise(X_outliers, m)

    indices = np.random.choice(len(X),10,replace=False)

    show_results(
    X,
    X_outliers,
    X_outlier_denoised,
    indices)

    show_results(
    X,
    X_low_noise,
    X_low_denoised,
    indices)

    show_results(
    X,
    X_high_noise,
    X_high_denoised,
    indices)

def show_results(original, noisy, denoised, indices):

    fig, axes = plt.subplots(len(indices), 3, figsize=(8, 20))

    for row, idx in enumerate(indices):

        axes[row,0].imshow(original[idx].reshape(16,16), cmap='gray')
        axes[row,0].set_title("Original")
        axes[row,0].axis('off')

        axes[row,1].imshow(noisy[idx].reshape(16,16), cmap='gray')
        axes[row,1].set_title("Noisy")
        axes[row,1].axis('off')

        axes[row,2].imshow(denoised[idx].reshape(16,16), cmap='gray')
        axes[row,2].set_title("Denoised")
        axes[row,2].axis('off')

    plt.tight_layout()
    plt.show()