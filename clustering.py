import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 
# Clustering libraries 
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, SpectralClustering
# Dimensionality reduction
from sklearn.manifold import MDS
# Internal validation metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 

# Constants 
plt.style.use('sciencemod.mplstyle')
directories = ['deep_recycle', 'deep_dropout', 'shallow_recycle', 'shallow_dropout']
labels = {'deep_recycle': 'Deep MSA + Recycle', 'deep_dropout': 'Deep MSA + Dropout', 'shallow_recycle': 'Shallow MSA + Recycle', 'shallow_dropout': 'Shallow MSA + Dropout' }
palette = sns.color_palette().as_hex()[0:4] 
colors = {'deep_recycle': palette[0], 'deep_dropout': palette[1], 'shallow_recycle': palette[2], 'shallow_dropout': palette[3]}
save_dir = './images/'
save_fmt = 'pdf' 

# Utility functions
def load_bundle() -> tuple[np.ndarray, np.ndarray]:
    """Load RMSD matrix and IDs from saved files."""
    data = np.load('./results/rmsd_matrix_all.npy', allow_pickle=True)
    ids = np.load('./results/ids_rmsd_matrix_all.npy', allow_pickle=True)
    return (data, ids)

# Agglomerative clustering
def agg_clust(mat: np.ndarray, mds: bool=False):
    if not mds: 
        model = AgglomerativeClustering(n_clusters=4, metric='precomputed', linkage='complete', compute_distances=True)
        labels = model.fit_predict(mat)
    else:
        return  
    print(pd.Series(labels).value_counts())
    return labels


if __name__ == "__main__":
    mat, ids = load_bundle() 
    print(np.unique(agg_clust(mat)))