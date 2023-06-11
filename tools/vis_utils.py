import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, OPTICS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import normalize
from hdbscan import HDBSCAN
from tqdm.notebook import tqdm
from yellowbrick.cluster import KElbowVisualizer
# from umap import UMAP


def apply_kmeans(data, n_clusters, max_iter=300):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter)
    clusters = kmeans.fit_predict(data)

    return clusters


def cluster_with_meanshift(data, bandwidth='auto', quantile=0.2, n_samples=2000, bin_seeding=True):

    if bandwidth == 'auto' or bandwidth == -1:
        bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=n_samples)
        print("Estimated Bandwidth is:", bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)

    ms.fit(data)

    return ms.labels_


def cluster_with_dbscan(data, eps=0.5, min_samples=5, metric='euclidean', leaf_size=30, scale_data=False):
    """
    :param data: (n_samples, n_dims) the data to be clustered
    :param eps:  threshold on the distance between two samples to be considered in the same neighborhood
    :param min_samples: min. number of samples in a neighborhood for a point to be considered as core point
    :param metric: [euclidean, cosine, cityblock, l1, l2, manhatten]
    :param leaf_size:
    :return: cluster labels, -1 denotes noisy samples
    """

    if scale_data:
        x = StandardScaler().fit_transform(data)
    else:
        x = data

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, leaf_size=leaf_size)

    db.fit(x)

    return db.labels_


def cluster_with_optics(data, min_samples, max_eps=1000, metric='euclidean', min_cluster_size=None):
    """
    :param data: (n_samples, n_dims) the data to be clustered
    :param min_samples: in. number of samples in a neighborhood for a point to be considered as core point
    :param max_eps
    :param metric: [euclidean, cosine, cityblock, l1, l2, manhatten, minkowski]
    :param min_cluster_size: minimum number of samples in a cluster, if None then min_samples will be used
    :return: cluster labels
    """

    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric, min_cluster_size=min_cluster_size)

    optics.fit(data)

    return optics.labels_


def cluster_with_hdbscan(data, min_samples, metric='euclidean', min_cluster_size=5, cluster_selection_epsilon=0.0,
                         algorithm='algorithm'):
    """
    :param data:
    :param min_samples:
    :param metric:
    :param min_cluster_size:
    :param algorithm:
    :param cluster_selection_epsilon:
    :return:
    """

    # A work around to the fact that cosine distance is not supported
    if metric == 'cosine':
        data = normalize(data, norm='l2')
        metric = 'euclidean'

    hdbscan = HDBSCAN(min_samples=min_samples, metric=metric, min_cluster_size=min_cluster_size, algorithm=algorithm,
                      cluster_selection_epsilon=cluster_selection_epsilon)

    hdbscan.fit(data)

    return hdbscan.labels_


def get_tsne(features, n_components=2):

    tsne = TSNE(n_components=n_components, random_state=0)
    tsne_features = tsne.fit_transform(features)

    return tsne_features


def get_pca(features, n_components):

    pca = PCA(n_components=n_components)
    features_scaled = MinMaxScaler().fit_transform(features)
    pca_features = pca.fit_transform(features_scaled)

    return pca_features, pca

# TODO: resolve Numba dependency issue in order to be able to use UMAP
# def get_umap(features, n_components, n_neighbors=15, min_dist=0.1):
#
#     umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
#
#     umap_features = umap.fit_transform(features)
#
#     return umap_features


def find_pca_n_components_for_variance_threshold(variance_ratio, threshold):

    var_index = 0
    var_sum = 0
    for v in variance_ratio:
        var_index += 1
        var_sum += v
        if var_sum >= threshold:
            break

    return var_index


def plot_bar(y, x=None, x_label='x', y_label='y', title=''):

    if x is None:
        x = np.arange(len(y))

    df = pd.DataFrame()
    df[x_label] = x
    df[y_label] = y
    fig = px.bar(
        df,
        x=x_label,
        y=y_label
    )
    fig.update_layout(title_text=title)

    return fig


def plot_line(x, y, x_label='x', y_label='y', markers=False, title=''):

    df = pd.DataFrame()
    df[x_label] = x
    df[y_label] = y
    fig = px.line(
        df,
        x=x_label,
        y=y_label,
        markers=markers
    )
    fig.update_layout(title_text=title)

    return fig


def plot_kmeans_clusters_tsne(data, cluster_mode='tsne', custom_embedding=None, default_n_clusters=50, len_threshold=8,
                              kmeans_max_iter=300):

    idx = np.where(data['lengths'] >= len_threshold)[0]
    reduced_df = data[data['lengths'] >= len_threshold]

    if cluster_mode == 'tsne':
        embedding = np.zeros((len(reduced_df['x_tsne']), 2))
        embedding[:, 0] = reduced_df['x_tsne']
        embedding[:, 1] = reduced_df['y_tsne']
    else:
        assert (custom_embedding is not None), "Custom Embedding must not be Null"
        assert len(custom_embedding.shape) == 2, "Custom Embedding must have dim=2"
        assert custom_embedding[idx].shape[0] == reduced_df.shape[0], "# of Embeddings must equal # of rows in data"
        embedding = custom_embedding[idx]

    def visualization(n_clusters=default_n_clusters):

        clusters = apply_kmeans(embedding, n_clusters, max_iter=kmeans_max_iter)
        fig = px.scatter(
            reduced_df,
            x='x_tsne',
            y='y_tsne',
            color=clusters,
            hover_data=['text'],
            color_continuous_scale=px.colors.qualitative.G10,
        )
        fig.update_layout(
            title_text=f'KMeans Clustering of Embeddings (#clusters = {n_clusters})'
        )
        return fig

    return visualization


def plot_meanshift_clusters_tsne(data, cluster_mode='tsne', custom_embedding=None, len_threshold=8):

    idx = np.where(data['lengths'] >= len_threshold)[0]
    reduced_df = data[data['lengths'] >= len_threshold]

    if cluster_mode == 'tsne':
        embedding = np.zeros((len(reduced_df['x_tsne']), 2))
        embedding[:, 0] = reduced_df['x_tsne']
        embedding[:, 1] = reduced_df['y_tsne']
    else:
        assert (custom_embedding is not None), "Custom Embedding must not be Null"
        assert len(custom_embedding.shape) == 2, "Custom Embedding must have dim=2"
        assert custom_embedding[idx].shape[0] == reduced_df.shape[0], "# of Embeddings must equal # of rows in data"
        embedding = custom_embedding[idx]

    def visualization(bandwidth='auto', quantile=0.2, n_samples=2000, bin_seeding=True):

        clusters = cluster_with_meanshift(embedding, bandwidth=bandwidth, quantile=quantile, n_samples=n_samples,
                                          bin_seeding=bin_seeding)
        num_clusters = len(np.unique(clusters))
        fig = px.scatter(
            reduced_df,
            x='x_tsne',
            y='y_tsne',
            color=clusters,
            hover_data=['text'],
            color_continuous_scale=px.colors.qualitative.G10,
        )
        fig.update_layout(
            title_text=f'MeanShift Clustering of Embeddings (#clusters = {num_clusters})'
        )
        return fig

    return visualization


def plot_dbscan_clusters_tsne(data, cluster_mode='tsne', custom_embedding=None, len_threshold=8):

    idx = np.where(data['lengths'] >= len_threshold)[0]
    reduced_df = data[data['lengths'] >= len_threshold]

    if cluster_mode == 'tsne':
        embedding = np.zeros((len(reduced_df['x_tsne']), 2))
        embedding[:, 0] = reduced_df['x_tsne']
        embedding[:, 1] = reduced_df['y_tsne']
    else:
        assert (custom_embedding is not None), "Custom Embedding must not be Null"
        assert len(custom_embedding.shape) == 2, "Custom Embedding must have dim=2"
        assert custom_embedding[idx].shape[0] == reduced_df.shape[0], "# of Embeddings must equal # of rows in data"
        embedding = custom_embedding[idx]

    def visualization(eps=0.5, min_samples=5, metric='euclidean', leaf_size=30, scale_data=False):

        clusters = cluster_with_dbscan(embedding, eps=eps, min_samples=min_samples, metric=metric, leaf_size=leaf_size,
                                       scale_data=scale_data)

        num_clusters = len(np.unique(clusters))
        num_outliers = np.sum(clusters == -1)

        fig = px.scatter(
            reduced_df,
            x='x_tsne',
            y='y_tsne',
            color=clusters,
            hover_data=['text'],
            color_continuous_scale=px.colors.qualitative.G10,
        )
        title = f'DBSCAN Clustering of Embeddings (#clusters = {num_clusters} with {metric} distance and {num_outliers} outliers)'
        fig.update_layout(
            title_text=title
        )

        return fig

    return visualization


def plot_optics_clusters_tsne(data, cluster_mode='tsne', custom_embedding=None, len_threshold=8):

    idx = np.where(data['lengths'] >= len_threshold)[0]
    reduced_df = data[data['lengths'] >= len_threshold]

    if cluster_mode == 'tsne':
        embedding = np.zeros((len(reduced_df['x_tsne']), 2))
        embedding[:, 0] = reduced_df['x_tsne']
        embedding[:, 1] = reduced_df['y_tsne']
    else:
        assert (custom_embedding is not None), "Custom Embedding must not be Null"
        assert len(custom_embedding.shape) == 2, "Custom Embedding must have dim=2"
        assert custom_embedding[idx].shape[0] == reduced_df.shape[0], "# of Embeddings must equal # of rows in data"
        embedding = custom_embedding[idx]

    def visualization(min_samples=5, max_eps=1000, metric='euclidean', min_cluster_size=-1):

        if min_cluster_size == -1:
            min_cluster_size = None

        clusters = cluster_with_optics(embedding, min_samples=min_samples, max_eps=max_eps, metric=metric,
                                       min_cluster_size=min_cluster_size)

        num_clusters = len(np.unique(clusters))
        num_outliers = np.sum(clusters == -1)

        fig = px.scatter(
            reduced_df,
            x='x_tsne',
            y='y_tsne',
            color=clusters,
            hover_data=['text'],
            color_continuous_scale=px.colors.qualitative.G10,
        )
        title = f'OPTICS Clustering of Embeddings (#clusters = {num_clusters} with {metric} distance and {num_outliers} outliers)'
        fig.update_layout(
            title_text=title
        )

        return fig

    return visualization


def plot_hdbscan_clusters_tsne(data, cluster_mode='tsne', custom_embedding=None, len_threshold=8):

    idx = np.where(data['lengths'] >= len_threshold)[0]
    reduced_df = data[data['lengths'] >= len_threshold]

    if cluster_mode == 'tsne':
        embedding = np.zeros((len(reduced_df['x_tsne']), 2))
        embedding[:, 0] = reduced_df['x_tsne']
        embedding[:, 1] = reduced_df['y_tsne']
    else:
        assert (custom_embedding is not None), "Custom Embedding must not be Null"
        assert len(custom_embedding.shape) == 2, "Custom Embedding must have dim=2"
        assert custom_embedding[idx].shape[0] == reduced_df.shape[0], "# of Embeddings must equal # of rows in data"
        embedding = custom_embedding[idx]

    def visualization(min_samples=5, metric='euclidean', min_cluster_size=-1, algorithm='best',
                      cluster_selection_epsilon=0.0):

        if min_cluster_size == -1:
            min_cluster_size = None

        clusters = cluster_with_hdbscan(embedding, min_samples=min_samples, metric=metric, algorithm=algorithm,
                                        min_cluster_size=min_cluster_size,
                                        cluster_selection_epsilon=cluster_selection_epsilon)

        num_clusters = len(np.unique(clusters))
        num_outliers = np.sum(clusters == -1)

        fig = px.scatter(
            reduced_df,
            x='x_tsne',
            y='y_tsne',
            color=clusters,
            hover_data=['text'],
            color_continuous_scale=px.colors.qualitative.G10,
        )
        title = f'HDBSCAN Clustering of Embeddings (#clusters = {num_clusters} with {metric} distance and {num_outliers} outliers)'
        fig.update_layout(
            title_text=title
        )

        return fig

    return visualization


def find_n_clusters_elbow_method(features, k_min, k_max):

    inertias = []
    ks = np.arange(k_min, k_max + 1)
    for k in tqdm(ks):

        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
        _ = kmeans.fit_predict(features)
        inertias.extend([kmeans.inertia_,])

    fig = plot_line(x=ks, y=inertias, x_label='n_clusters', y_label='intertia', markers=True, title='Inertia vs n_clusters')

    return fig


def yellow_brick_elbow_method(features, k_min, k_max):

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(k_min, k_max))

    visualizer.fit(features)
    visualizer.show()
    return visualizer


def yellow_brick_silhouette_method(features, k_min, k_max):

    model = KMeans()
    visualizer = KElbowVisualizer(model, metric='silhouette', k=(k_min, k_max))

    visualizer.fit(features)
    visualizer.show()
    return visualizer
