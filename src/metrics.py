import numpy as np
import matplotlib.pyplot as plt


def gammaidx(X, k):
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    sorted_dist = np.sort(dist_matrix, axis=1)
    k_nearest = sorted_dist[:, 1:k+1]
    y = np.mean(k_nearest, axis=1)
    return y


def auc(y_true, y_pred, plot=False):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    P = np.sum(y_true == 1)
    N = np.sum(y_true == -1)
    
    if P == 0 or N == 0:
        raise ValueError("y_true must contain both -1 and +1 labels.")

    idx = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[idx]
    
    tprs = [0.0]
    fprs = [0.0]
    
    tp_count = 0
    fp_count = 0
    auc_val = 0.0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp_count += 1
        else:
            
            fp_count += 1
            auc_val += tp_count / P
            
        tprs.append(tp_count / P)
        fprs.append(fp_count / N)
        
    c = auc_val / N

    if plot:
        plt.figure(figsize=(6, 5))
        plt.plot(fprs, tprs, color='darkorange', lw=2, label=f'ROC curve (area = {c:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

    return c
