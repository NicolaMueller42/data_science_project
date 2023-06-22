import streamlit as st
import time
import numpy as np
import pandas as pd
import code.description_data as data
import requests
from code.visualisation import get_embeddings, compute_tsne, compute_kpca, \
    plot_2d, plot_3d, get_hover_data, add_industry_clusters, add_clusters
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(layout="wide")


def close_expander():
    st.session_state.expand_input = False


@st.cache_data
def compute_clustering(test_description, embedding_type, method, n_clusters, dimensions):
    embedding_type = "max" if embedding_type == "Maximum of Features" else \
        ("mean" if embedding_type == "Mean of Features" else embedding_type)
    embeddings = get_embeddings(embedding_type, test_description)
    if method == "t-SNE":
        clusters, projected = compute_tsne(n_clusters, dimensions, embeddings)
    elif method == "Kernel PCA":
        clusters, projected = compute_kpca(n_clusters, dimensions, embeddings)
    return clusters, projected


def get_lat_lon_of_request(search_string):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(search_string) + '?format=json'
    response = requests.get(url).json()
    return response[0]["lat"], response[0]["lon"]


@st.cache_data
def load_map_df(labels):
    data = np.zeros((len(labels), 2))
    for i, company in enumerate(labels):
        print(i, company)
        try:
            lat, lon = get_lat_lon_of_request(company + ", Saarland")
        except Exception as e:
            print(e)
            continue
        data[i, 0] = lat
        data[i, 1] = lon
    coords_df = pd.DataFrame(data, columns=["latitude", "longitude"])
    return pd.merge(pd.DataFrame(labels, columns=["labels"]), coords_df, right_index=True, left_index=True)

if "expand_input" not in st.session_state:
    st.session_state.expand_input = True
with st.expander("Input", expanded=st.session_state.expand_input):
    with st.form("Search"):
        test_label = st.text_input("Enter company name:")
        test_description = st.text_input("Enter your company description:")
        embedding_type = st.radio("Select embedding",
                                  options=["Maximum of Features", "Mean of Features", "Concatenated Features"],
                                  help="Select how text data will be embedded. "
                                       "Maximum of Features: Takes the maximum of feature embeddings "
                                       "across sentences in the description.\n"
                                       "Mean of Features: Takes the mean of feature embeddings "
                                       "across sentences in the description.\n"
                                       "Concatenated Features: Concatenates all features into a large input feature. "
                                       "Computation usually takes longer using this method."
                                  )
        method = st.radio("Select dimensionality reduction method", options=["t-SNE", "Kernel PCA"])
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=24, value=10)
        dimensions = st.radio("Select number of dimensions to use for clustering", options=[2, 3])
        submit = st.form_submit_button("Submit", on_click=close_expander)
if submit and test_description:
    clusters, projected = compute_clustering(test_description, embedding_type, method, n_clusters, dimensions)
    with st.expander("Competitors", expanded=True):
        df = get_hover_data()
        competitors = df[clusters[:-1] == clusters[-1]]
        st.write(f"Your competitors are:")
        st.dataframe(competitors)
        st.write(f"Your competitors are working in the following industries: {np.unique(competitors['Industry'].values)}")
        if dimensions == 2:
            figure = plot_2d(clusters[:-1], projected[:-1])
            figure = add_clusters(figure, clusters[:-1], projected[:-1])
            figure.add_trace(go.Scatter(
                x=[projected[-1, 0]],
                y=[projected[-1, 1]],
                marker=dict(
                    color=px.colors.qualitative.Light24[clusters[-1]],
                    size=20,
                    line=dict(
                        color='White',
                        width=2
                    ),),
                mode="markers+text",
                text=test_label,
                hoverinfo="none",
                textposition="top center",
                showlegend=False
            ))
            figure.update_layout(height=400)
            st.plotly_chart(figure, use_container_width=True)
        else:
            figure = plot_3d(clusters[:-1], projected[:-1])
            figure.add_trace(go.Scatter3d(
                x=[projected[-1, 0]],
                y=[projected[-1, 1]],
                z=[projected[-1, 2]],
                marker=dict(
                    color=px.colors.qualitative.Light24[clusters[-1]],
                    size=12,
                    line=dict(
                        color='White',
                        width=2
                    ), ),
                mode="markers+text",
                hoverinfo="none",
                text=test_label,
                textposition="top center",
                showlegend=False
            ))
            figure.update_layout(height=400)
            st.plotly_chart(figure, use_container_width=True)
elif submit and not test_description:
    st.warning("Description must not be empty or else your company cannot be clustered!")