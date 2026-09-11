# Unsupervised Learning: K-Means + PCA

## Task
This project applies an unsupervised learning pipeline to a real dataset using K-Means clustering and PCA.

## Dataset
The dataset is the classic Iris flower measurements dataset. The original species labels were intentionally removed, so clustering is performed without using labels.

Features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Method
1. Load the unlabeled dataset.
2. Standardize all numeric features with `StandardScaler`.
3. Test K-Means for k = 2 through 8.
4. Compare inertia (elbow method) and silhouette scores.
5. Select **k = 3**, supported by the elbow behavior and the meaningful three-group structure.
6. Fit final K-Means and save cluster assignments.
7. Apply PCA to reduce the four standardized features to two dimensions.
8. Plot the PCA projection colored by K-Means cluster.
9. Report PCA explained variance.

## Results
- Selected clusters: **k = 3**
- PC1 explained variance: **72.96%**
- PC2 explained variance: **22.85%**
- Total variance captured by the 2D PCA projection: **95.81%**
- Silhouette score at k=3: **0.4599**

## Real-world interpretation
The three discovered clusters represent groups of Iris flowers with similar measurement profiles. In particular, petal length and petal width contribute strongly to separating the groups, while the sepal measurements provide additional variation. Because clustering was performed without species labels, the algorithm discovered these groups from feature similarity rather than being told the correct species.

## Files
- `iris_unlabeled.csv` — input data with no labels
- `clustered_iris.csv` — data with K-Means cluster assignments
- `unsupervised_learning.py` — complete reproducible analysis
- `requirements.txt` — Python dependencies
- `outputs/elbow_plot.png` — elbow analysis
- `outputs/silhouette_plot.png` — silhouette analysis
- `outputs/pca_clusters.png` — required PCA 2D cluster visualization
- `outputs/pca_explained_variance.csv` — PCA variance ratios
- `outputs/cluster_feature_means.csv` — average feature values per cluster
- `outputs/k_selection_metrics.csv` — k-selection metrics

## How to run

```bash
pip install -r requirements.txt
python unsupervised_learning.py
```

The script regenerates the analysis and plots inside the `outputs` folder.
