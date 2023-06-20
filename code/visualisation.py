from code.clustering_tsne import project_tsne, predict_clusters
from code.clustering_kpca import predict_cluster, fit_kernel_pca, cluster_data
from code.embeddings import get_description_embeddings
from code.description_data import train_descriptions, train_labels
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from paths import ECO_CSV
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull
from scipy import interpolate


@st.cache_data
def get_hover_data():
    df = pd.read_csv(ECO_CSV)
    return df


@st.cache_resource
def get_embeddings(embedding_type="max"):
    return get_description_embeddings(train_descriptions, embed_type=embedding_type)


@st.cache_data
def compute_hull(points):
    if len(points) == 2:
        # create fake points to compute convex hull
        if abs(points[0, 0] - points[1, 0]) < 1:
            # same x
            fake_point_1 = np.array(
                [np.mean(points[:, 0]), points[0, 1] + (np.max(points[:, 0] - np.mean(points[:, 0])))])
            fake_point_2 = np.array(
                [np.mean(points[:, 0]), points[0, 1] - (np.max(points[:, 0] - np.mean(points[:, 0])))])
        elif abs(points[0, 1] - points[1, 1]) < 1:
            # same y
            fake_point_1 = np.array(
                [points[0, 0] + (np.max(points[:, 1] - np.mean(points[:, 1]))), np.mean(points[:, 1])])
            fake_point_2 = np.array(
                [points[0, 0] - (np.max(points[:, 1] - np.mean(points[:, 1]))), np.mean(points[:, 1])])
        else:
            fake_point_1 = np.array([np.min(points[:, 0]), np.min(points[:, 1])])
            fake_point_2 = np.array([np.max(points[:, 0]), np.max(points[:, 1])])
        points = np.append(points, fake_point_1).reshape(-1, 2)
        points = np.append(points, fake_point_2).reshape(-1, 2)
    elif len(points) == 1:
        return points[:, 0], points[:, 1]

    # compute convex hull
    try:
        hull = ConvexHull(points)  # computation might not work if data points in a cluster have a very weird position
    except:
        return points[:, 0], points[:, 1]
    x_hull = np.append(points[hull.vertices, 0],
                       points[hull.vertices, 0][0])
    y_hull = np.append(points[hull.vertices, 1],
                       points[hull.vertices, 1][0])

    # interpolate to get a smooth shape
    dist = np.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    return interp_x, interp_y


@st.cache_data
def compute_tsne(n_clusters, dimensions, train_embeddings):
    # Compute the clustering of the training data
    projected_train_embeddings = project_tsne(data=train_embeddings, dimensions=dimensions,
                                              perplexity=np.sqrt(len(train_embeddings)),
                                              random_state=42)
    clusters, projected = predict_clusters(projected_data=projected_train_embeddings, n_clusters=n_clusters)
    return clusters, projected


@st.cache_data
def compute_kpca(n_clusters, dimensions, train_embeddings):
    kpca = fit_kernel_pca(train_embeddings, n_components=dimensions)
    clustering = cluster_data(train_embeddings, kpca, n_clusters)
    return predict_cluster(train_embeddings, kpca, clustering)


@st.cache_data
def plot_2d(clusters, projected):
    str_cluster = [str(cluster) for cluster in clusters]
    df = pd.merge(
        pd.DataFrame(projected, columns=["x", "y"]),
        get_hover_data(),
        right_index=True, left_index=True
    )
    fig = px.scatter(df,
                     x="x",
                     y="y",
                     color=str_cluster,
                     labels={"color": "Cluster"},
                     category_orders={"color": list(sorted(str_cluster, key=lambda x: int(x)))},
                     color_discrete_sequence=px.colors.qualitative.Light24,
                     text="Company",
                     hover_name="Company",
                     custom_data=df,
                     hover_data={
                         "x": False,
                         "y": False,
                         "Industry": True,
                         "Products": True,
                         "Customer Base": True,
                         "Market Positioning": True,
                         "Revenue": True})
    fig.update_traces(textposition='top center',
                      hovertemplate='<b>%{customdata[3]}</b><br>'
                                    'Industry: %{customdata[4]} <br>'
                                    'Products: %{customdata[5]} <br>'
                                    'Customer base: %{customdata[6]} <br>'
                                    'Market Position: %{customdata[7]} <br>'
                                    'Revenue: %{customdata[8]}€')
    fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    return fig


@st.cache_data
def plot_3d(clusters, projected):
    str_cluster = [str(cluster) for cluster in clusters]
    df = pd.merge(
        pd.DataFrame(projected),
        get_hover_data(),
        right_index=True, left_index=True
    )
    fig = px.scatter_3d(df,
                        x=0,
                        y=1,
                        z=2,
                        color=str_cluster,
                        labels={"color": "Cluster"},
                        custom_data=df,
                        category_orders={"color": list(sorted(str_cluster, key=lambda x: int(x)))},
                        color_discrete_sequence=px.colors.qualitative.Light24,
                        text=train_labels,
                        hover_name="Company",
                        hover_data={
                            "Industry": True,
                            "Products": True,
                            "Customer Base": True,
                            "Market Positioning": True,
                            "Revenue": True})

    fig.update_traces(textposition='top center',
                      hovertemplate='<b>%{customdata[4]}</b><br>'
                                    'Industry: %{customdata[5]} <br>'
                                    'Products: %{customdata[6]} <br>'
                                    'Customer base: %{customdata[7]} <br>'
                                    'Market Position: %{customdata[8]} <br>'
                                    'Revenue: %{customdata[9]}€',)
    fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    return fig


@st.cache_data
def add_clusters(fig, clusters, projected):
    traces = []
    for cluster in np.unique(clusters):
        points = projected[clusters == cluster]
        cluster_x, cluster_y = compute_hull(points)
        mean_x = np.mean(cluster_x)
        mean_y = np.mean(cluster_y)
        traces.append(go.Scatter(
            x=cluster_x,
            y=cluster_y,
            fill="toself",
            fillcolor=px.colors.qualitative.Light24[cluster],
            opacity=0.1,
            mode="lines",
            line=dict(color=px.colors.qualitative.Light24[cluster]),
            text=f"Cluster {cluster}",
            textposition="middle center",
            showlegend=False,
            hoverinfo="none"
        ))
        traces.append(go.Scatter(
            x=[mean_x],
            y=[mean_y],
            mode="text",
            fillcolor=px.colors.qualitative.Light24[cluster],
            text=f"Cluster {cluster}",
            textposition="middle center" if len(points[:, 0]) > 1 else "bottom right",
            textfont=dict(color=px.colors.qualitative.Light24[cluster]),
            showlegend=False,
            hoverinfo="none"
        ))
    new_fig = go.Figure(data=traces)
    new_fig.add_traces(data=fig.data)
    new_fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    new_fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=True, mirror=True, showline=True)
    new_fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=True, mirror=True, showline=True)
    return new_fig


@st.cache_data
def add_industry_clusters(fig, projected):
    df = pd.merge(
        pd.DataFrame(projected),
        get_hover_data(),
        right_index=True, left_index=True
    )
    traces = []
    for i, industry in enumerate(np.unique(df["Industry"])):
        points = projected[df["Industry"] == industry]
        cluster_x, cluster_y = compute_hull(points)
        mean_x = np.mean(cluster_x)
        mean_y = np.mean(cluster_y)
        traces.append(go.Scatter(
            x=cluster_x,
            y=cluster_y,
            fill="toself",
            fillcolor=px.colors.qualitative.Light24[i],
            opacity=0.1,
            mode="lines",
            line=dict(color=px.colors.qualitative.Light24[i]),
            text=f"{industry}",
            textposition="middle center",
            showlegend=False,
        ))
        traces.append(go.Scatter(
            x=[mean_x],
            y=[mean_y],
            mode="text",
            fillcolor=px.colors.qualitative.Light24[i],
            text=f"{industry}",
            textposition="middle center" if len(points[:, 0]) > 1 else "bottom right",
            textfont=dict(color=px.colors.qualitative.Light24[i]),
            showlegend=False,
            hoverinfo="none"
        ))
    new_fig = go.Figure(data=traces)
    new_fig.add_traces(data=fig.data)
    new_fig.update_layout(
        height=750,
        hoverlabel=dict(
            font_size=16
        ),
        margin=go.Margin(
            l=0,
            r=0,
            b=0,
            t=10
        )
    )
    return new_fig
