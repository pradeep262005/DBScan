import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm

def generate_dbscan_plot(eps, min_samples, output_path='static/dbscan_result.png'):
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    df = pd.DataFrame(X, columns=['X', 'Y'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['X', 'Y']])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    df['Cluster'] = labels

    unique_labels = np.unique(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 6))

    for label, color in zip(unique_labels, colors):
        cluster_data = df[df['Cluster'] == label]
        if label == -1:
            plt.scatter(cluster_data['X'], cluster_data['Y'],
                        facecolors='black', edgecolors='k', label='Noise', s=60)
        else:
            plt.scatter(cluster_data['X'], cluster_data['Y'],
                        facecolors=color, edgecolors='k', label=f'Cluster {label}', s=60)

    plt.legend()
    plt.title(f'DBSCAN Result (eps={eps}, min_samples={min_samples})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
