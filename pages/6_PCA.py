import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

st.title("Principal Component Analysis")

st.subheader('Theory')
on = st.toggle(">")

## Writing Theory 
if on:
    st.write("""Principal Component Analysis is a dimensionality reduction.
            It is common in ML world to work with very large number of features
            to model predictions.
            But, high dimensions can lead to some complex challenges.  
             """)
    
    st.write("""PCA helps in 2 ways:""")
    
    st.write("""1. It allows us to graphically visualise our feature space
    and hence providing useful insights.""")

    st.write("""2. It reduces the length of feature vector in our model. 
            Making it accessible for traditional ML techniques.""")


st.divider()

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X -  self.mean

        # covariance, functions needs samples as columns
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # projects data
        X = X - self.mean
        return np.dot(X, self.components.T)

data = datasets.load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.scatter(
    x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
fig.colorbar(im, ax=ax)

st.pyplot(fig)

