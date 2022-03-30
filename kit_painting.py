import numpy as np
from sklearn.decomposition import PCA


def scatter_pca(data, label, ax, pca=None):
    # PCA
    if pca is None:
        pca_dim = 2
        pca = PCA(n_components=pca_dim)
        pca_data = pca.fit_transform(data)
        pca_msg = "Variance explained by first {} principal components: {}".format(pca_dim, pca.explained_variance_ratio_)
    else:
        pca_data = pca.transform(data)
        pca_msg = "PCA unchanged"
    print(pca_msg)

    # plot
    for cat in range(max(label) + 1):
        ptlb = "cat{}".format(cat)
        ax.scatter(*np.where(label == cat, pca_data.T, None), label=ptlb)
    ax.legend()

    return pca, pca_data



