import numpy as np
from sklearn.decomposition import PCA


class PaintPCA:
    def __init__(self, data_ref):
        pca_dim = 2
        self.pca = PCA(n_components=pca_dim)
        self.pca_data_ref = self.pca.fit_transform(data_ref)
        pca_msg = "Variance explained by first {} principal components: {}".format(pca_dim, self.pca.explained_variance_ratio_)
        print(pca_msg)

    def scatter(self, ax, label, data_new=None):
        if data_new is None:
            pca_data = self.pca_data_ref
        else:
            pca_data = self.pca.transform(data_new)

        for cat in range(max(label) + 1):
            ax.scatter(*np.where(label == cat, pca_data.T, None), label=cat)


