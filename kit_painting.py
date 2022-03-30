import numpy as np
from sklearn.decomposition import PCA

def photo_pca_truth(data, label, ax):
    # PCA
    pca_dim = 2
    pca = PCA(n_components=pca_dim)
    pca_xy = pca.fit_transform(data)
    pca_msg = "Variance explained by first {} principal components: {}".format(pca_dim, pca.explained_variance_ratio_)
    print(pca_msg)

    # plot
    for cat in range(max(label) + 1):
        ptlb = "cat{}".format(cat)
        ax.scatter(*np.where(label == cat, pca_xy.T, None), label=ptlb)
    ax.set_title("true data")
    ax.legend()

    return pca, pca_xy
