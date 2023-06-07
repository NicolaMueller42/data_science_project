"""
This script contains all the functions necessary for clustering data.
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import plotly.express as px

# Prints the results of clustering
def print_clusters(data_labels, clustering_labels):
    clusters = {}
    for c_label in clustering_labels:
        clusters[c_label] = []

    for e_label, c_label in zip(data_labels, clustering_labels):
        clusters[c_label].append(e_label)

    for c_label in clusters.keys():
        print("Cluster " + str(c_label) + ":")
        print(clusters[c_label])
        print("\n")

# Fits an instance of Principal Components Analysis using given data
def fit_pca(data, dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(data)

    return pca

# Fits an instance of Kernel Principal Components Analysis using given data
def fit_kernel_pca(data, dimensions, kernel='poly', gamma=0.05):
    kpca = KernelPCA(n_components=dimensions, kernel=kernel, gamma=gamma)
    kpca.fit(data)

    return kpca

# Computes the clustering: First the data's dimensionality is reduced by transforming it according to the given PCA/KPCA
# instance and then K-Means is applied to cluster the proejcted data
def cluster_data(data, pca, n_clusters, n_init=50, init="random", random_state=69):
    projected_data = pca.transform(data)
    clustering = KMeans(n_clusters=n_clusters, n_init=n_init, init=init, random_state=random_state).fit(projected_data)

    return clustering

# Assigns new data points to established clusters
def predict_cluster(data, pca, clustering):
    projected_data = pca.transform(data)
    clustering_labels = clustering.predict(projected_data)

    return clustering_labels, projected_data

# Visualizes the results of clustering as 2D or 3D scatter plots
def visualize_clustering(train_clusters, train_labels, train_projected, test_clusters=None, test_labels=None,
                         test_projected=None, fontsize=5, cmap="rainbow", dimensions=2):
    if dimensions == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title("Clustering of description embeddings projected onto 2D space using KPCA")

        ax.scatter(train_projected[:, 0], train_projected[:, 1], c=train_clusters, cmap=cmap)
        for i, txt in enumerate(train_labels):
            ax.text(train_projected[i, 0], train_projected[i, 1], txt, fontsize=fontsize)

        if test_labels is not None:
            ax.scatter(test_projected[:, 0], test_projected[:, 1], c=test_clusters, cmap=cmap)
            for i, txt in enumerate(test_labels):
                ax.text(test_projected[i, 0], test_projected[i, 1], txt, fontsize=fontsize)
    else:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title("Clustering of description embeddings projected onto 3D space using KPCA")

        ax.scatter(train_projected[:, 0], train_projected[:, 1], train_projected[:, 2], c=train_clusters, cmap=cmap)
        for i, txt in enumerate(train_labels):
            ax.text(train_projected[i, 0], train_projected[i, 1], train_projected[i, 2], txt, fontsize=fontsize)

        if test_labels is not None:
            ax.scatter(test_projected[:, 0], test_projected[:, 1], test_projected[:, 2], c=test_clusters, cmap=cmap)
            for i, txt in enumerate(test_labels):
                ax.text(test_projected[i, 0], test_projected[i, 1], test_projected[i, 2], txt, fontsize=fontsize)

    plt.show(block=True)

# Does the same thing as the function above, but returns the plot as a plotly plot
def get_clustering_plot(train_clusters, train_labels, train_projected, test_clusters=None, test_labels=None,
                         test_projected=None, fontsize=5, cmap="rainbow", dimensions=2):
    if dimensions == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title("Clustering of description embeddings projected onto 2D space using KPCA")

        ax.scatter(train_projected[:, 0], train_projected[:, 1], c=train_clusters, cmap=cmap)
        for i, txt in enumerate(train_labels):
            ax.text(train_projected[i, 0], train_projected[i, 1], txt, fontsize=fontsize)

        if test_labels is not None:
            ax.scatter(test_projected[:, 0], test_projected[:, 1], c=test_clusters, cmap=cmap)
            for i, txt in enumerate(test_labels):
                ax.text(test_projected[i, 0], test_projected[i, 1], txt, fontsize=fontsize)
    else:
        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(projection='3d')
        ax.set_title("Clustering of description embeddings projected onto 3D space using KPCA")

        ax.scatter(train_projected[:, 0], train_projected[:, 1], train_projected[:, 2], c=train_clusters, cmap=cmap)
        for i, txt in enumerate(train_labels):
            ax.text(train_projected[i, 0], train_projected[i, 1], train_projected[i, 2], txt, fontsize=fontsize)

        if test_labels is not None:
            ax.scatter(test_projected[:, 0], test_projected[:, 1], test_projected[:, 2], c=test_clusters, cmap=cmap)
            for i, txt in enumerate(test_labels):
                ax.text(test_projected[i, 0], test_projected[i, 1], test_projected[i, 2], txt, fontsize=fontsize)
    return fig

# Projects data using t-SNE and visualizes the results as 2D or 3D scatter plots
def visualize_tsne(data, labels, dimensions, perplexity, fontsize=5, random_state=69):
    if dimensions == 2:
        projected = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(np.array(data))

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title("Description embeddings projected onto 2D space using t-SNE")

        ax.scatter(projected[:, 0], projected[:, 1])
        for i, txt in enumerate(labels):
            ax.text(projected[i, 0], projected[i, 1], txt, fontsize=fontsize)

        plt.show(block=True)
    elif dimensions == 3:
        projected = TSNE(n_components=3, perplexity=perplexity, random_state=random_state).fit_transform(np.array(data))

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2])
        for i, txt in enumerate(labels):
            ax.text(projected[i, 0], projected[i, 1], projected[i, 2], txt, fontsize=fontsize)

        plt.show(block=True)

# Does the same thing as the function above, but returns the plots instead of showing them
def get_tSNE_plots(data, labels, dimensions, perplexity, fontsize=5, random_state=69):
    if dimensions == 2:
        projected = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(np.array(data))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title("Description embeddings projected onto 2D space using t-SNE")

        ax.scatter(projected[:, 0], projected[:, 1])
        for i, txt in enumerate(labels):
            ax.text(projected[i, 0], projected[i, 1], txt, fontsize=fontsize)

        plt.show(block=True)
    elif dimensions == 3:
        projected = TSNE(n_components=3, perplexity=perplexity, random_state=random_state).fit_transform(np.array(data))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2])
        for i, txt in enumerate(labels):
            ax.text(projected[i, 0], projected[i, 1], projected[i, 2], txt, fontsize=fontsize)

    return fig
