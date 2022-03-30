import numpy as np
from sklearn.decomposition import PCA


class PaintPCA:
    def __init__(self, data_ref):
        pca_dim = 2
        self.pca = PCA(n_components=pca_dim)
        self.pca_data_ref = self.pca.fit_transform(data_ref)
        pca_msg = "Variance explained by first {} principal components: {}".format(pca_dim, self.pca.explained_variance_ratio_)
        print(pca_msg)

    def scatter(self, ax, cat_label, data_new=None):
        if data_new is None:
            pca_data = self.pca_data_ref
        else:
            pca_data = self.pca.transform(data_new)

        for cat in range(max(cat_label) + 1):
            ax.scatter(*np.where(cat_label == cat, pca_data.T, None), label=cat)

    def tripole(self, ax, s_pole, n_pole):
        """
        input: multiple pairs of s_pole and n_pole
        """
        pca_s, pca_n = [self.pca.transform(pole) for pole in (s_pole, n_pole)]
        pca_c = (pca_s + pca_n) / 2
        for ps, pc, pn in zip(pca_s, pca_c, pca_n):
            c = next(ax._get_lines.prop_cycler)['color']
            ax.plot(*np.transpose([ps, pn]), color=c)
            ax.scatter(*pc, marker='o', s=200, facecolors='none', linewidths=2, edgecolors=c)
