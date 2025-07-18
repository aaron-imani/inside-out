from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# from tsnecuda import TSNE


class Reduction:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        # self.tsne = TSNE(n_components=2, perplexity=10)
        self.tsne = TSNE(perplexity=5, random_state=42)

    def fit(self, X):
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

    def get_tsne(self, X):
        standardized_data = StandardScaler().fit_transform(X)
        return self.tsne.fit_transform(standardized_data)
