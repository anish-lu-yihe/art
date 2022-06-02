import numpy as np

import seaborn as sns
from kit_handy import *
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


class VisualiseReplay:
    def __init__(self, data_ref):
        self.data_ref = np.array(data_ref)
        self.data_dim = self.data_ref.shape[1]

    def distance_to_truth(self, ax_hist, replay):
        norm_l1 = least_l1_to_data(replay, self.data_ref) / self.data_dim
        dist_mean = np.mean(norm_l1)
        dist_median = np.median(norm_l1)
        dist_95tile = np.percentile(norm_l1, 95)

        if ax_hist is not None:
            sns.kdeplot(y=norm_l1, fill=True, bw_adjust=.1, clip=[0, 1], ax=ax_hist)
            [ax_hist.axhline(d, ls=s, color='k') for d, s in zip((dist_median, dist_mean, dist_95tile), ('solid', 'dashdot', 'dotted'))]
        return dist_mean, dist_median, dist_95tile

    def property_of_memory(self, ax, FuzzyART):
        cat_num = FuzzyART.w.shape[0]
        cat_size = FuzzyART._getSize()
        size_mean = np.mean(cat_size)
        size_median = np.median(cat_size)
        point_cat_idx = np.flatnonzero(cat_size)
        point_cat_num = cat_num - point_cat_idx.size

        if ax is not None:
            bin_num = 3
            bin_uplim = np.max(cat_size) / 2 ** np.arange(bin_num - 1, -1, -1)
            bin_idx = np.digitize(cat_size[point_cat_idx], bins=bin_uplim, right=True)
            bin_count = np.unique(bin_idx, return_counts=True)
            bar_count = np.zeros(bin_num + 1, dtype=int)
            bar_count[bin_count[0] + 1] = bin_count[1]
            bar_count[0] = point_cat_num
            bar_label = np.insert(bin_uplim, 0, 0)
            ax.barh(np.arange(bin_num + 1), bar_count, alpha=.4)
            ax.set_yticks(np.arange(bin_num + 1))
            ax.set_yticklabels(bar_label.round(2))
            ax.set_xticks(np.arange(np.max(bar_count) + 1))
        return cat_num, point_cat_num, size_mean, size_median



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
            transformed_data = self.pca.transform(data)
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
                try:
                    cathull = ConvexHull(pca_cv)
                except:
                    print('Convex hull cannot be found due to linear dependence of all points.')
                hullxy = np.append(cathull.vertices, cathull.vertices[0])
                ax.plot(pca_cv[hullxy, 0], pca_cv[hullxy, 1], '--')

