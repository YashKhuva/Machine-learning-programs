import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score


# ================= SINGLE LINKAGE =================

# Load the breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform agglomerative clustering
agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='single')
cluster_labels = agg_clustering.fit_predict(X_scaled)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Single Linkage Dendrogram')

print("Single Linkage Performance:")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")

plt.tight_layout()
plt.show()


# ================= AVERAGE LINKAGE =================

# Load dataset
cancer = load_breast_cancer()
X = cancer.data

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Clustering
agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='average')
cluster_labels = agg_clustering.fit_predict(X_scaled)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Average Linkage Dendrogram')

print("\nAverage Linkage Performance:")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")

plt.tight_layout()
plt.show()


# ================= COMPLETE LINKAGE =================

# Load dataset
cancer = load_breast_cancer()
X = cancer.data

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Clustering
agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete')
cluster_labels = agg_clustering.fit_predict(X_scaled)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Complete Linkage Dendrogram')

print("\nComplete Linkage Performance:")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")

plt.tight_layout()
plt.show()
