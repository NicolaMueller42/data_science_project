"""
This script is used to find the optimal hyperparameters for PCA/KPCA via grid search.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA, KernelPCA
from code.description_data import train_descriptions
from code.embeddings import get_description_embeddings, get_full_embeddings

embeddings = get_full_embeddings(train_descriptions)
def my_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)

param_grid = [{
    "gamma": np.linspace(0.03, 0.05, 10),
    "kernel": ["rbf", "sigmoid", "linear", "poly"],
    "degree": [2, 3, 4, 5]
}]
kpca = KernelPCA(fit_inverse_transform=True, n_components=3)
grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=my_scorer, n_jobs=-1)
grid_search.fit(embeddings)
print(grid_search.cv_results_)
print(grid_search.best_params_)
