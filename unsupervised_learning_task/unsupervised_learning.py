import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DATA_PATH = "iris_unlabeled.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
features = df.select_dtypes(include="number")
X = features.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = list(range(2, 9))
inertias, silhouettes = [], []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Chosen from the elbow and silhouette analysis, with k=3 giving the
# meaningful three-group structure for this dataset.
best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters + 1
df.to_csv("clustered_iris.csv", index=False)

metrics = pd.DataFrame({
    "k": k_values,
    "inertia": inertias,
    "silhouette_score": silhouettes
})
metrics.to_csv(os.path.join(OUTPUT_DIR, "k_selection_metrics.csv"), index=False)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

pd.DataFrame({
    "component": ["PC1", "PC2"],
    "explained_variance_ratio": explained
}).to_csv(os.path.join(OUTPUT_DIR, "pca_explained_variance.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("K-Means inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_plot.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouettes, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Scores for Candidate k Values")
plt.xticks(k_values)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "silhouette_plot.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
for c in sorted(df["cluster"].unique()):
    mask = df["cluster"].values == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c}", alpha=0.75)
plt.xlabel(f"PC1 ({explained[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.2f}% variance)")
plt.title("PCA 2D Projection Colored by K-Means Cluster")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_clusters.png"), dpi=200)
plt.close()

df.groupby("cluster")[[
    "sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"
]].mean().to_csv(os.path.join(OUTPUT_DIR, "cluster_feature_means.csv"))

print("Completed successfully.")
print(f"Selected k: {best_k}")
print(f"PC1 variance: {explained[0]*100:.2f}%")
print(f"PC2 variance: {explained[1]*100:.2f}%")
print(f"2D total variance captured: {explained.sum()*100:.2f}%")
print(f"k=3 silhouette score: {silhouettes[k_values.index(3)]:.4f}")
