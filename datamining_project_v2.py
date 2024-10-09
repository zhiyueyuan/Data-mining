import os
os.chdir('E:/DataMining_Renew/datamining_project')




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('combined_data.csv')

# Select features
features = ['PM2.5_Avg_2019', '2021_AIC', 'Death Rate', 'Urban population']

# Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# 4. PCA Dimensionality Reduction
pca = PCA(n_components=2)  
X_pca = pca.fit_transform(X_scaled)

# Output the proportion of variance explained by PCA
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio by first two components: {explained_variance}")

# Visualizing Explained Variance
plt.figure(figsize=(8, 6))
plt.bar(['PC1', 'PC2'], explained_variance, color='skyblue')
plt.title('Explained Variance Ratio by PCA Components')
plt.ylabel('Variance Ratio')
plt.tight_layout()
plt.show()

# 5. K-means
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca) 


data['Cluster'] = clusters

# Visualization of clustering results
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, linewidths=3)  # 聚类中心
plt.title('K-means Clustering with PCA')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar()
plt.show()

# Clustering results analysis
for i in range(kmeans.n_clusters):
    print(f"Countries in Cluster {i}:")
    print(data[data['Cluster'] == i]['Location'].tolist())

# Cluster center analysis
cluster_centers = kmeans.cluster_centers_
centers_original_space = pca.inverse_transform(cluster_centers)  # 将簇中心转换回原始特征空间
centers_df = pd.DataFrame(scaler.inverse_transform(centers_original_space), columns=features)
print("Cluster Centers in Original Feature Space:")
print(centers_df)

# Radar chart showing the characteristic distribution of cluster centers
categories = features
N = len(categories)

for i in range(kmeans.n_clusters):
    cluster_center = centers_df.iloc[i].values
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    cluster_center = np.concatenate((cluster_center, [cluster_center[0]]))
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, cluster_center, color='blue', alpha=0.25)
    ax.plot(angles, cluster_center, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(f"Cluster {i} Feature Distribution", size=15)
    plt.show()

# The number of countries included in each cluster
cluster_counts = data['Cluster'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
plt.bar(cluster_counts.index, cluster_counts.values, color='orange')
plt.title('Number of Countries in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Countries')
plt.xticks(cluster_counts.index)
plt.tight_layout()
plt.show()

# Random Forest Feature Importance Analysis

X = data[features]
y = data['Cluster']

# Divide the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


feature_importances = rf_model.feature_importances_

# Visualize feature importance
indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Random Forest")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center", color='green')
plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.tight_layout()
plt.show()


for i, v in enumerate(feature_importances[indices]):
    print(f"Feature: {features[indices[i]]}, Importance: {v:.4f}")