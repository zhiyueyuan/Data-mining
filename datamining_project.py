import os
# 更改工作目录
#os.chdir('E:/DataMining_Renew/datamining_project')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    features = ['PM2.5_Avg_2019', 'PM2.5_Change_2010_2019', 
                '2021 [YR2021]_9020000:ACTUAL INDIVIDUAL CONSUMPTION', 
                'Mean Growth 2017-2021_9020000:ACTUAL INDIVIDUAL CONSUMPTION',
                'Death Rate', 'Mean Growth Rate (%)',
                'Urban population (% of total population)', 'Mean Growth (2012-2022)']
    
    X = data[features]
    
    # Use KNN imputation for missing values
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    
    return data, X_scaled

def perform_clustering(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans.cluster_centers_

def perform_pca(X):
    pca = PCA()
    pca_result = pca.fit_transform(X)
    
    # Select number of components that explain 90% of variance
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1
    
    return pca_result[:, :n_components], pca.explained_variance_ratio_[:n_components]

def plot_clusters(X_pca, clusters, centers):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.colorbar(scatter)
    plt.title('Country Clusters based on PCA')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

def plot_correlation_heatmap(X):
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Features')
    plt.show()

def perform_feature_importance(X, y):
    # Feature selection
    selector = SelectKBest(f_classif, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Random Forest with cross-validation
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X_selected, y, cv=5)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Train final model
    rf.fit(X_selected, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def main():
    data, X_scaled = load_and_preprocess_data('Data-mining\preprocessed_data\combined_data.csv')
    
    # Clustering
    clusters, centers = perform_clustering(X_scaled)
    data['Cluster'] = clusters
    
    # PCA
    X_pca, explained_var_ratio = perform_pca(X_scaled)
    print(f"Explained variance ratio: {explained_var_ratio}")
    
    # Plotting
    plot_clusters(X_pca, clusters, centers)
    plot_correlation_heatmap(X_scaled)
    
    # Feature Importance
    y = (data['PM2.5_Avg_2019'] > data['PM2.5_Avg_2019'].quantile(0.8)).astype(int)
    feature_importance = perform_feature_importance(X_scaled, y)
    print("Feature Importance:")
    print(feature_importance)
    
    # Top countries analysis
    top_countries = data.sort_values('PM2.5_Avg_2019', ascending=False).head(10)
    print("\nTop 10 Countries with Highest PM2.5 Levels:")
    print(top_countries[['Location', 'PM2.5_Avg_2019', '2021 [YR2021]_9020000:ACTUAL INDIVIDUAL CONSUMPTION', 'Urban population (% of total population)']])

if __name__ == "__main__":
    main()