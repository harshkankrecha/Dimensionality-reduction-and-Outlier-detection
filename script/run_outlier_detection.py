from src.metrics import gammaidx,auc
import matplotlib.pyplot as plt
import numpy as np


def outliers_calc():
    banana = np.load('banana.npz')
    X = banana['data']
    y = banana['label'].flatten()

    if X.shape[0]==2:X=X.T
    
    positive_class = np.unique(y)[0]
    X_pos = X[y == positive_class]

    print("Positive samples:", X_pos.shape)

    contamination_rates = [0.01, 0.10, 0.50, 1.00]

    results = {
        'gamma_k3': [],
        'gamma_k10': [],
        'mean_distance': []
    }

    num_inliers = X_pos.shape[0]

    for rate in contamination_rates:

        auc_k3 = []
        auc_k10 = []
        auc_mean = []

        num_inliers = X_pos.shape[0]

    for rate in contamination_rates:

        auc_k3 = []
        auc_k10 = []
        auc_mean = []

    num_outliers = int(rate * num_inliers)

    for trial in range(100):

        
        X_out = np.random.uniform(-4, 4, size=(num_outliers, 2))

        
        X_total = np.vstack([X_pos, X_out])

        y_true = np.hstack([
            -1 * np.ones(num_inliers),
            +1 * np.ones(num_outliers)
        ])

        
        score_k3 = gammaidx(X_total, k=3)

        
        score_k10 = gammaidx(X_total, k=10)

        
        score_mean = distance_to_mean(X_total)

        
        auc1 = auc(y_true, score_k3)
        auc2 = auc(y_true, score_k10)
        auc3 = auc(y_true, score_mean)

        auc_k3.append(auc1)
        auc_k10.append(auc2)
        auc_mean.append(auc3)

        results['gamma_k3'].append(auc_k3)
        results['gamma_k10'].append(auc_k10)
        results['mean_distance'].append(auc_mean)


    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    method_names = ['gamma_k3', 'gamma_k10', 'mean_distance']

    for i, rate in enumerate(contamination_rates):

        data = [
            results['gamma_k3'][i],
            results['gamma_k10'][i],
            results['mean_distance'][i]
        ]

        axes[i].boxplot(data, labels=['k=3', 'k=10', 'mean'])

        axes[i].set_title(f'Contamination = {int(rate*100)}%')
        axes[i].set_ylabel('AUC')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def distance_to_mean(X):
    mean = np.mean(X, axis=0)
    dist = np.sqrt(np.sum((X - mean) ** 2, axis=1))
    return dist

def outliers_disp():
    banana = np.load('banana.npz')
    X = banana['data']
    y = banana['label'].flatten()

    if X.shape[0]==2:X=X.T
    
    positive_class = np.unique(y)[0]
    X_pos = X[y == positive_class]
    rate = 0.10
    num_inliers = X_pos.shape[0]

    num_outliers = int(rate * num_inliers)

    X_out = np.random.uniform(-4, 4, size=(num_outliers, 2))

    X_total = np.vstack([X_pos, X_out])

    labels = np.hstack([
        np.zeros(num_inliers),
        np.ones(num_outliers)
    ])

    score_k3 = gammaidx(X_total, k=3)
    score_k10 = gammaidx(X_total, k=10)
    score_mean = distance_to_mean(X_total)

    methods = [
        ('Gamma k=3', score_k3),
        ('Gamma k=10', score_k10),
        ('Distance to Mean', score_mean)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, scores) in zip(axes, methods):

        sizes = 20 + 200 * (scores - scores.min()) / (scores.max() - scores.min())

        ax.scatter(
            X_total[labels == 0][:, 0],
            X_total[labels == 0][:, 1],
            s=sizes[labels == 0],
            c='blue',
            alpha=0.6,
            label='Inliers'
        )

        ax.scatter(
            X_total[labels == 1][:, 0],
            X_total[labels == 1][:, 1],
            s=sizes[labels == 1],
            c='red',
            alpha=0.8,
            label='Outliers'
        )
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()