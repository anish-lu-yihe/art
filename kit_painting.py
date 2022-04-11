import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


class PaintPCA:
    def __init__(self, data_ref):
        self.noPCA = np.shape(data_ref)[1] == 2
        if self.noPCA:
            pca_msg = "No PCA performed"
            self.pca_data_ref = np.array(data_ref)
        else:
            pca_dim = 2
            self.pca = PCA(n_components=pca_dim)
            self.pca_data_ref = self.pca.fit_transform(data_ref)
            pca_msg = "Variance explained by first {} principal components: {}".format(pca_dim, self.pca.explained_variance_ratio_)

        print(pca_msg)

    def _transform(self, data):
        if self.noPCA:
            transformed_data = data
        else:
            transformed_data = self.pca.fit_transform(data)
        return transformed_data

    def scatter(self, ax, cat_label, data_new=None):
        if data_new is None:
            pca_data = self.pca_data_ref
        else:
            pca_data = self._transform(data_new)

        for cat in range(max(cat_label) + 1):
            ax.scatter(*np.where(cat_label == cat, pca_data.T, None), label=cat)
        ax.scatter(*np.where(cat_label == -1, pca_data.T, None), color='k', marker='^', label=-1)

    def tripole(self, ax, s_pole, n_pole):
        """
        input: multiple categorical pairs of s_pole and n_pole
        """
        pca_s, pca_n = [self._transform(pole) for pole in (s_pole, n_pole)]
        pca_c = (pca_s + pca_n) / 2
        for ps, pc, pn in zip(pca_s, pca_c, pca_n):
            c = next(ax._get_lines.prop_cycler)['color']
            ax.plot(*np.transpose([ps, pn]), color=c)
            ax.scatter(*pc, marker='o', s=200, facecolors='none', linewidths=2, edgecolors=c)

    def boundary(self, ax, vertices):
        """
        input multiple categorical groups of vertices
        """
        for catidx, catvert in enumerate(vertices):
            if np.allclose(catvert[0], catvert[1], atol=1e-15):
                print("Category {} is too small".format(catidx))
            else:
                pca_cv = self._transform(catvert)
                cathull = ConvexHull(pca_cv)
                hullxy = np.append(cathull.vertices, cathull.vertices[0])
                ax.plot(pca_cv[hullxy, 0], pca_cv[hullxy, 1], '--')

