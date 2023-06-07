import streamlit as st
import numpy as np
import mpld3
import streamlit.components.v1 as components
from code.description_data import train_descriptions, train_labels, test_descriptions, test_labels
from code.embeddings import get_description_embeddings
from code.clustering import fit_pca, fit_kernel_pca, cluster_data, predict_cluster, \
    get_clustering_plot, get_tSNE_plots, print_clusters

st.title("Company Clustering")
train_embeddings = get_description_embeddings(train_descriptions, max=True)
test_embeddings = get_description_embeddings(test_descriptions, max=True)

n_clusters = int(np.floor(np.sqrt(len(train_descriptions)))) + 4

pca = fit_kernel_pca(data=train_embeddings, dimensions=3)

clustering = cluster_data(data=train_embeddings, pca=pca, n_clusters=n_clusters, n_init=50,
                          init="random", random_state=69)

train_clusters, train_projected = predict_cluster(data=train_embeddings, pca=pca, clustering=clustering)
test_clusters, test_projected = predict_cluster(data=test_embeddings, pca=pca, clustering=clustering)

print_clusters(train_labels, train_clusters)

fig = get_clustering_plot(train_clusters=train_clusters, train_labels=train_labels, train_projected=train_projected,
                     test_clusters=test_clusters, test_labels=test_labels, test_projected=test_projected,
                     fontsize=8, cmap="rainbow", dimensions=3)

st.pyplot(fig)

fig = get_tSNE_plots(data=train_embeddings + test_embeddings, labels=train_labels + test_labels, dimensions=2,
                     perplexity=np.sqrt(len(train_embeddings) + len(test_embeddings)), fontsize=5, random_state=69)

st.pyplot(fig)
