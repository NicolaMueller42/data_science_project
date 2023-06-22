import streamlit as st
import numpy as np
import requests
import os

import paths
from code.description_data import train_labels
import urllib
import pandas as pd
from code.visualisation import get_embeddings, compute_tsne, compute_kpca, \
    plot_2d, plot_3d, plot_map, add_clusters


st.set_page_config(layout="wide")


@st.cache_data
def compute_clustering(embedding_type, method, n_clusters):
    embedding_type = "max" if embedding_type == "Maximum of Features" else \
        ("mean" if embedding_type == "Mean of Features" else embedding_type)
    embeddings = get_embeddings(embedding_type)
    if method == "t-SNE":
        clusters_2d, projected_2d = compute_tsne(n_clusters, 2, embeddings)
        clusters_3d, projected_3d = compute_tsne(n_clusters, 3, embeddings)
    elif method == "Kernel PCA":
        clusters_2d, projected_2d = compute_kpca(n_clusters, 2, embeddings)
        clusters_3d, projected_3d = compute_kpca(n_clusters, 3, embeddings)
    return {"2d": (clusters_2d, projected_2d),
            "3d": (clusters_3d, projected_3d)}


selection_area, plot_area = st.columns([1, 4])
with selection_area:
    with st.form("Clustering parameters"):
        embedding_type = st.radio("Select embedding",
                                  options=["Maximum of Features", "Mean of Features", "Concatenated Features"],
                                  help="Select how text data will be embedded. "
                                       "Maximum of Features: Takes the maximum of feature embeddings "
                                       "across sentences in the description.\n"
                                       "Mean of Features: Takes the mean of feature embeddings "
                                       "across sentences in the description.\n"
                                       "Concatenated Features: Concatenates all features into a large input feature. "
                                       "Computation usually takes longer using this method.")
        method = st.radio("Select dimensionality reduction method", options=["t-SNE", "Kernel PCA"])
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=24, value=10)
        st.form_submit_button("Compute clustering")
    plot_data_dict = compute_clustering(embedding_type, method, n_clusters)
    fig_2d = plot_2d(plot_data_dict["2d"][0], plot_data_dict["2d"][1])
    fig_3d = plot_3d(plot_data_dict["3d"][0], plot_data_dict["3d"][1])
    dimensions = st.radio("Select number of dimensions to display", options=["2d", "3d"])
with plot_area:
    if dimensions == "2d":
        compare_mode = st.radio("Plot Style", options=["Cluster Plot", "Map", "Both"], horizontal=True)
        if compare_mode == "Map":
            new_fig = plot_map(clusters=plot_data_dict["2d"][0])
            st.plotly_chart(new_fig, use_container_width=True)
            st.warning("Some companies might be missing on this map!")
        elif compare_mode == "Cluster Plot":
            new_fig = add_clusters(fig_2d, plot_data_dict["2d"][0], plot_data_dict["2d"][1])
            st.plotly_chart(new_fig, use_container_width=True)
        else:
            plot, map_col = st.columns([1, 1])
            with plot:
                new_fig = add_clusters(fig_2d, plot_data_dict["2d"][0], plot_data_dict["2d"][1])
                st.plotly_chart(new_fig, use_container_width=True)
            with map_col:
                new_fig = plot_map(clusters=plot_data_dict["2d"][0])
                st.plotly_chart(new_fig, use_container_width=True)
                st.warning("Some companies might be missing on this map!")
    else:
        st.plotly_chart(fig_3d, use_container_width=True)

