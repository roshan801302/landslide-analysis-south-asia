#!/usr/bin/env python3
"""
Landslide-Event Analysis, Spatial Clustering of Hotspots, and Fatality Classification in South Asia using Machine Learning

Title: Landslide-Event Analysis, Spatial Clustering of Hotspots, and Fatality Classification in South Asia using Machine Learning
Alignment: SDG 11 (Sustainable Cities and Communities) and SDG 15 (Life on Land)

This script performs:
1. Data Acquisition & Loading - NASA Global Landslide Catalog (GLC) with synthetic fallback
2. Data Cleaning & Preprocessing - Pipeline with imputation, scaling, encoding
3. Unsupervised Spatial Clustering - K-means/Hierarchical clustering with Silhouette analysis
4. Supervised Classification - Fatal vs Non-fatal prediction using 4 classifiers
5. Evaluation & Reporting - Comprehensive metrics with ROC curves

Author: Senior ML Engineer (Research Implementation)
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

# Set style for visualizations
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def install_instructions():
    """Print installation instructions for required libraries."""
    print("=" * 80)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 80)
    print("To install the required libraries, run the following command:")
    print("\npip install pandas numpy scikit-learn matplotlib seaborn")
    print("\nOr with conda:")
    print("\nconda install pandas numpy scikit-learn matplotlib seaborn")
    print("=" * 80)
    print()


def load_landslide_data(file_path=None):
    """
    Load NASA Global Landslide Catalog (GLC) dataset or create synthetic fallback.
    
    Args:
        file_path (str, optional): Path to the CSV file. If None, looks in common locations.
    
    Returns:
        pd.DataFrame: Loaded landslide dataset
    """
    print("=" * 80)
    print("1. DATA ACQUISITION & LOADING")
    print("=" * 80)
    
    # Define potential file paths
    potential_paths = [
        file_path,
        "/home/roshan/Pictures/ml project/Global_Landslide_Catalog_Export_rows.csv",
        "./Global_Landslide_Catalog_Export_rows.csv",
        "../Global_Landslide_Catalog_Export_rows.csv",
        "data/Global_Landslide_Catalog_Export_rows.csv",
    ]
    
    df = None
    for path in potential_paths:
        if path is not None and os.path.exists(path):
            try:
                print(f"Loading dataset from: {path}")
                df = pd.read_csv(path)
                print(f"✓ Successfully loaded dataset with {len(df)} records and {len(df.columns)} columns")
                break
            except Exception as e:
                print(f"✗ Error loading from {path}: {e}")
    
    if df is None:
        print("\n⚠ File not found locally. Creating realistic synthetic fallback dataframe...")
        print("This mimics NASA GLC schema for South Asian context (Western Ghats & Himalayas)")
        df = create_synthetic_data()
        print(f"✓ Created synthetic dataset with {len(df)} records")
    
    # Display basic info
    print("\nDataset Overview:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {', '.join(df.columns[:10])}...")
    print(f"  - Date range: {df['event_date'].min() if 'event_date' in df else 'N/A'} to {df['event_date'].max() if 'event_date' in df else 'N/A'}")
    
    return df


def create_synthetic_data(n_samples=1500):
    """
    Create realistic synthetic landslide data mimicking NASA GLC schema for South Asia.
    
    Args:
        n_samples (int): Number of synthetic samples to generate
    
    Returns:
        pd.DataFrame: Synthetic dataset
    """
    np.random.seed(42)
    
    # South Asia bounds: Western Ghats and Himalayas focus
    # Himalayas: lat ~25-35°N, lon ~70-95°E (includes India, Nepal, Bhutan)
    # Western Ghats: lat ~8-20°N, lon ~72-78°E (India)
    # Also include parts of Pakistan, Bangladesh, Sri Lanka
    
    data = []
    
    # Define regions
    regions = [
        # Western Ghats (India)
        {'country': 'India', 'lat_range': (8, 20), 'lon_range': (72, 78), 'weight': 0.35},
        # Himalayas - India
        {'country': 'India', 'lat_range': (25, 35), 'lon_range': (75, 95), 'weight': 0.25},
        # Himalayas - Nepal
        {'country': 'Nepal', 'lat_range': (26, 30), 'lon_range': (80, 88), 'weight': 0.15},
        # Himalayas - Pakistan
        {'country': 'Pakistan', 'lat_range': (30, 36), 'lon_range': (70, 77), 'weight': 0.08},
        # Bangladesh (hill tracts)
        {'country': 'Bangladesh', 'lat_range': (22, 26), 'lon_range': (91, 93), 'weight': 0.05},
        # Sri Lanka (highlands)
        {'country': 'Sri Lanka', 'lat_range': (6, 9), 'lon_range': (80, 81), 'weight': 0.05},
        # Bhutan
        {'country': 'Bhutan', 'lat_range': (27, 28), 'lon_range': (89, 92), 'weight': 0.07},
    ]
    
    # Trigger types common in South Asia
    triggers = ['rain', 'downpour', 'monsoon', 'continuous_rain', 'tropical_cyclone', 'earthquake']
    trigger_probs = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
    
    for i in range(n_samples):
        # Select region
        region = np.random.choice(regions, p=[r['weight'] for r in regions])
        lat = np.random.uniform(region['lat_range'][0], region['lat_range'][1])
        lon = np.random.uniform(region['lon_range'][0], region['lon_range'][1])
        
        # Generate date (2000-2020)
        year = np.random.randint(2000, 2021)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        event_date = f"{month:02d}/{day:02d}/{year} 12:00:00 AM"
        
        # Generate trigger
        trigger = np.random.choice(triggers, p=trigger_probs)
        
        # Fatalities - skewed (mostly non-fatal)
        if np.random.random() < 0.25:  # 25% fatal
            fatalities = np.random.randint(1, 51)
        else:
            fatalities = 0
        
        data.append({
            'event_id': i + 1000,
            'event_date': event_date,
            'country_name': region['country'],
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'landslide_trigger': trigger,
            'fatality_count': fatalities,
            'injury_count': np.random.randint(0, 20) if fatalities > 0 else np.random.randint(0, 5),
            'landslide_size': np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.35, 0.15]),
            'landslide_category': 'landslide',
            'source_name': 'Synthetic',
        })
    
    return pd.DataFrame(data)


def filter_south_asia(df):
    """
    Filter dataset for South Asian / Indian events focusing on Western Ghats and Himalayas.
    
    Args:
        df (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Filtered South Asia dataset
    """
    print("\n" + "=" * 80)
    print("FILTERING FOR SOUTH ASIA (Western Ghats & Himalayas)")
    print("=" * 80)
    
    # Define South Asian countries
    sa_countries = ['India', 'Nepal', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Bhutan']
    
    # Filter by country
    if 'country_name' in df.columns:
        df_filtered = df[df['country_name'].isin(sa_countries)].copy()
    else:
        # If no country column, filter by coordinates
        df_filtered = df.copy()
    
    print(f"Records after country filter: {len(df_filtered)}")
    
    # Filter by rainfall-triggered events (as specified)
    rainfall_triggers = ['rain', 'downpour', 'monsoon', 'continuous_rain', 'tropical_cyclone', 
                        'snowfall_snowmelt', 'Rain', 'rainfall']
    
    if 'landslide_trigger' in df.columns:
        # Create mask for rainfall triggers
        trigger_mask = df_filtered['landslide_trigger'].str.lower().isin(
            [t.lower() for t in rainfall_triggers]
        ) if df_filtered['landslide_trigger'].dtype == 'object' else False
        
        # Also include if contains 'rain' or 'monsoon'
        rain_mask = df_filtered['landslide_trigger'].str.lower().str.contains('rain|monsoon|downpour|cyclone', 
                                                                              na=False) if df_filtered['landslide_trigger'].dtype == 'object' else False
        df_filtered = df_filtered[trigger_mask | rain_mask].copy()
        print(f"Records after rainfall trigger filter: {len(df_filtered)}")
    
    # Focus on Western Ghats and Himalayan bounds
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Western Ghats: lat ~8-20°N, lon ~72-78°E
        # Himalayas: lat ~25-36°N, lon ~70-95°E
        wg_mask = (df_filtered['latitude'].between(8, 20)) & (df_filtered['longitude'].between(72, 78))
        him_mask = (df_filtered['latitude'].between(25, 36)) & (df_filtered['longitude'].between(70, 95))
        combined_mask = wg_mask | him_mask
        df_filtered = df_filtered[combined_mask].copy()
        print(f"Records after Western Ghats & Himalayas bounds filter: {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("\n⚠ No records after filtering. Using all South Asian records from original subset.")
        df_filtered = df[df['country_name'].isin(sa_countries)].copy() if 'country_name' in df.columns else df.copy()
    
    print(f"✓ Final filtered dataset: {len(df_filtered)} records")
    print(f"  - Countries: {df_filtered['country_name'].value_counts().to_dict()}")
    
    return df_filtered


def clean_and_preprocess(df):
    """
    Data cleaning & preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Filtered dataset
    
    Returns:
        tuple: (processed_df, X, y, preprocessor)
    """
    print("\n" + "=" * 80)
    print("2. DATA CLEANING & PREPROCESSING")
    print("=" * 80)
    
    df_clean = df.copy()
    
    # Handle missing values in key columns
    print("Handling missing values...")
    key_cols = ['latitude', 'longitude', 'event_date', 'landslide_trigger', 'fatality_count']
    for col in key_cols:
        if col in df_clean.columns:
            missing = df_clean[col].isna().sum()
            if missing > 0:
                print(f"  - {col}: {missing} missing values")
    
    # Drop rows with missing critical coordinates
    if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
        before = len(df_clean)
        df_clean.dropna(subset=['latitude', 'longitude'], inplace=True)
        print(f"  ✓ Dropped {before - len(df_clean)} rows with missing coordinates")
    
    # Fill missing fatality_count with 0
    if 'fatality_count' in df_clean.columns:
        df_clean['fatality_count'] = pd.to_numeric(df_clean['fatality_count'], errors='coerce').fillna(0)
        print(f"  ✓ Filled missing fatalities with 0")
    
    # Fill missing trigger with 'unknown'
    if 'landslide_trigger' in df_clean.columns:
        df_clean['landslide_trigger'] = df_clean['landslide_trigger'].fillna('unknown')
        print(f"  ✓ Filled missing triggers with 'unknown'")
    
    # Create binary target column for classification: fatal_class = 1 if fatalities > 0, else 0
    print("\nCreating target variable...")
    df_clean['fatal_class'] = (df_clean['fatality_count'] > 0).astype(int)
    class_dist = df_clean['fatal_class'].value_counts()
    print(f"  - Fatal (1): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(df_clean)*100:.2f}%)")
    print(f"  - Non-fatal (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(df_clean)*100:.2f}%)")
    
    # Extract temporal features (month, season) from dates
    print("\nExtracting temporal features...")
    if 'event_date' in df_clean.columns:
        try:
            # Parse dates - handle multiple formats
            df_clean['event_date_parsed'] = pd.to_datetime(df_clean['event_date'], errors='coerce', format='%m/%d/%Y %I:%M:%S %p')
            # Try alternative format if needed
            mask_na = df_clean['event_date_parsed'].isna()
            if mask_na.sum() > 0:
                df_clean.loc[mask_na, 'event_date_parsed'] = pd.to_datetime(df_clean.loc[mask_na, 'event_date'], errors='coerce')
            
            df_clean['month'] = df_clean['event_date_parsed'].dt.month.fillna(0).astype(int)
            df_clean['year'] = df_clean['event_date_parsed'].dt.year.fillna(0).astype(int)
            
            # Define seasons (Northern Hemisphere: India/Nepal/Pakistan/Bhutan/Bangladesh)
            def get_season(month):
                if pd.isna(month) or month == 0:
                    return 'unknown'
                if month in [12, 1, 2]:
                    return 'winter'
                elif month in [3, 4, 5]:
                    return 'spring'
                elif month in [6, 7, 8, 9]:  # Monsoon dominant
                    return 'monsoon'
                elif month in [10, 11]:
                    return 'post_monsoon'
                return 'unknown'
            
            df_clean['season'] = df_clean['month'].apply(get_season)
            print(f"  ✓ Extracted month, year, and season")
            print(f"    - Season distribution: {df_clean['season'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  ✗ Error parsing dates: {e}")
            df_clean['month'] = 0
            df_clean['season'] = 'unknown'
    else:
        df_clean['month'] = 0
        df_clean['season'] = 'unknown'
    
    # Prepare features for modeling
    print("\nPreparing feature sets...")
    
    # Features for clustering (spatial only)
    X_cluster = df_clean[['latitude', 'longitude']].copy()
    
    # Features for classification
    # Use relevant features: coordinates, month, season, trigger
    feature_cols = []
    if 'latitude' in df_clean.columns:
        feature_cols.append('latitude')
    if 'longitude' in df_clean.columns:
        feature_cols.append('longitude')
    if 'month' in df_clean.columns:
        feature_cols.append('month')
    if 'season' in df_clean.columns:
        feature_cols.append('season')
    if 'landslide_trigger' in df_clean.columns:
        feature_cols.append('landslide_trigger')
    
    X_class = df_clean[feature_cols].copy()
    y_class = df_clean['fatal_class'].copy()
    
    print(f"  - Clustering features: {list(X_cluster.columns)}")
    print(f"  - Classification features: {list(X_class.columns)}")
    
    # Build scikit-learn Pipeline for classification (median imputation, scaling, one-hot encoding)
    print("\nBuilding preprocessing pipeline to prevent data leakage...")
    
    numeric_features = ['latitude', 'longitude', 'month']
    categorical_features = ['season', 'landslide_trigger']
    
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'
    )
    
    print("  ✓ Pipeline components:")
    print("    - Numeric: median imputation → standard scaling")
    print("    - Categorical: constant imputation → one-hot encoding")
    
    return df_clean, X_cluster, X_class, y_class, preprocessor


def perform_spatial_clustering(X_cluster, df_clean, save_plot=True):
    """
    Perform unsupervised spatial clustering for hotspot identification.
    
    Args:
        X_cluster (pd.DataFrame): Latitude/longitude coordinates
        df_clean (pd.DataFrame): Cleaned dataframe
        save_plot (bool): Whether to save the plot
    
    Returns:
        tuple: (kmeans_model, cluster_labels, optimal_k)
    """
    print("\n" + "=" * 80)
    print("3. UNSUPERVISED SPATIAL CLUSTERING (HOTSPOT IDENTIFICATION)")
    print("=" * 80)
    
    if len(X_cluster) < 2:
        print("⚠ Insufficient data for clustering")
        return None, np.array([]), 2
    
    # Evaluate silhouette scores for k = 2 to 6
    print("Evaluating Silhouette Score for k = 2 to 6...")
    silhouette_scores = []
    k_range = range(2, 7)  # k from 2 to 6
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels_k = kmeans.fit_predict(X_cluster)
        sil_score = silhouette_score(X_cluster, cluster_labels_k)
        silhouette_scores.append(sil_score)
        print(f"  k={k}: Silhouette Score = {sil_score:.4f}")
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    max_score = max(silhouette_scores)
    
    print(f"\n✓ Optimal k determined: k={optimal_k} (Silhouette Score = {max_score:.4f})")
    print("  This helps separate hazard zones (Himalayan belt vs Western Ghats)")
    
    # Perform final clustering with optimal k
    print(f"\nPerforming K-means clustering with k={optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_cluster)
    
    # Also try hierarchical clustering for comparison
    print("\nPerforming Hierarchical Clustering (Agglomerative) for comparison...")
    hier = AgglomerativeClustering(n_clusters=optimal_k)
    hier_labels = hier.fit_predict(X_cluster)
    hier_sil = silhouette_score(X_cluster, hier_labels)
    print(f"  Hierarchical Clustering Silhouette Score (k={optimal_k}): {hier_sil:.4f}")
    
    # Add labels to dataframe
    df_clean['cluster_kmeans'] = cluster_labels
    df_clean['cluster_hierarchical'] = hier_labels
    
    # Generate and save scatter plot
    if save_plot:
        print("\nGenerating spatial cluster visualization...")
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_cluster['longitude'], X_cluster['latitude'], 
                             c=cluster_labels, cmap='tab10', s=50, alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title(f'Spatial Clustering of Landslide Hotspots (K-means, k={optimal_k})\nSouth Asia: Western Ghats & Himalayas', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add region annotations
        plt.text(75, 15, 'Western Ghats', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.text(82, 30, 'Himalayas', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plot_path = 'landslide_spatial_clusters.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved cluster plot to: {plot_path}")
        plt.close()
    
    # Cluster statistics
    print("\nCluster Statistics (K-means):")
    for i in range(optimal_k):
        cluster_mask = cluster_labels == i
        n_points = cluster_mask.sum()
        mean_lat = X_cluster[cluster_mask]['latitude'].mean()
        mean_lon = X_cluster[cluster_mask]['longitude'].mean()
        fatalities = df_clean[cluster_mask]['fatality_count'].sum()
        fatal_events = df_clean[cluster_mask]['fatal_class'].sum()
        print(f"  Cluster {i}: {n_points} events | Center: ({mean_lat:.2f}, {mean_lon:.2f}) | Total fatalities: {fatalities} | Fatal events: {fatal_events}")
    
    return kmeans_final, cluster_labels, optimal_k, silhouette_scores


def train_classification_models(X, y, preprocessor, test_size=0.2, random_state=42):
    """
    Train and evaluate four classifiers for fatal vs non-fatal prediction.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        preprocessor (ColumnTransformer): Preprocessing pipeline
        test_size (float): Test set size
        random_state (int): Random state
    
    Returns:
        dict: Dictionary containing trained models, predictions, and metrics
    """
    print("\n" + "=" * 80)
    print("4. SUPERVISED CLASSIFICATION (FATAL VS. NON-FATAL)")
    print("=" * 80)
    
    # Split data chronologically or via train_test_split (80/20) as specified
    print(f"Splitting data into train/test sets ({int((1-test_size)*100)}/{int(test_size*100)})...")
    
    # Try chronological split if dates available
    if 'event_date_parsed' in X.index or 'year' in X.columns:
        # Use random split as specified (80/20) for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Test set: {len(X_test)} samples")
    print(f"  - Class distribution in train: {y_train.value_counts().to_dict()}")
    
    # Define classifiers with class_weight='balanced' for class imbalance
    classifiers = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', 
            max_iter=2000, 
            random_state=random_state,
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', 
            random_state=random_state,
            max_depth=10
        ),
        'k-NN': KNeighborsClassifier(n_neighbors=5),
        'Gaussian Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    print("\nTraining classifiers...")
    for name, clf in classifiers.items():
        print(f"  - Training {name}...")
        
        # Create pipeline with preprocessing and classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        
        # Calculate metrics (emphasizing minority fatal class)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
        
        results[name] = {
            'pipeline': pipeline,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"    ✓ Done - F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, ROC-AUC={roc_auc:.4f}")
    
    return results, X_test, y_test


def plot_roc_curves(results, y_test, save_plot=True):
    """
    Plot ROC curves for all models.
    
    Args:
        results (dict): Classification results
        y_test (pd.Series): True labels
        save_plot (bool): Whether to save plot
    """
    print("\nGenerating ROC curves...")
    plt.figure(figsize=(10, 8))
    
    for name, res in results.items():
        if res['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {res["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves for Fatal Landslide Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plot_path = 'roc_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves to: {plot_path}")
    plt.close()


def evaluate_and_report(results, y_test, X_test):
    """
    Print comprehensive evaluation metrics and summary table.
    
    Args:
        results (dict): Classification results
        y_test (pd.Series): True labels
        X_test (pd.DataFrame): Test features
    """
    print("\n" + "=" * 80)
    print("5. EVALUATION & REPORTING")
    print("=" * 80)
    
    print("\n⚠ IMPORTANT: Since missing a fatal landslide is dangerous,")
    print("   we emphasize Recall (Sensitivity) for the fatal class (minority).")
    print()
    
    # Detailed report for each model
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(classification_report(y_test, res['y_pred'], 
                                   target_names=['Non-Fatal (0)', 'Fatal (1)'],
                                   zero_division=0))
        print(f"Confusion Matrix:\n{res['confusion_matrix']}")
        print(f"ROC-AUC: {res['roc_auc']:.4f}")
    
    # Summary table (formatted for paper)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - Model Comparison (Copy to Research Paper)")
    print("=" * 80)
    
    # Create comparison dataframe
    comparison_data = []
    for name, res in results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{res['accuracy']:.4f}",
            'Precision (Fatal)': f"{res['precision']:.4f}",
            'Recall (Fatal)': f"{res['recall']:.4f}",
            'F1-Score (Fatal)': f"{res['f1']:.4f}",
            'ROC-AUC': f"{res['roc_auc']:.4f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Find best model by recall (critical for fatal landslides)
    best_recall_idx = np.argmax([r['recall'] for r in results.values()])
    best_recall_name = list(results.keys())[best_recall_idx]
    best_recall_val = list(results.values())[best_recall_idx]['recall']
    
    best_f1_idx = np.argmax([r['f1'] for r in results.values()])
    best_f1_name = list(results.keys())[best_f1_idx]
    best_f1_val = list(results.values())[best_f1_idx]['f1']
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"• Best model by Recall (critical for safety): {best_recall_name} (Recall={best_recall_val:.4f})")
    print(f"• Best model by F1-Score: {best_f1_name} (F1={best_f1_val:.4f})")
    print(f"• Project Alignment: SDG 11 (Sustainable Cities & Communities), SDG 15 (Life on Land)")
    print("=" * 80)


def main():
    """Main execution function for the complete pipeline."""
    start_time = datetime.now()
    
    # Print banner
    print("=" * 80)
    print("LANDSLIDE-EVENT ANALYSIS, SPATIAL CLUSTERING & FATALITY CLASSIFICATION")
    print("South Asia - Western Ghats & Himalayas")
    print(f"Execution started: {start_time}")
    print("=" * 80)
    
    # Install instructions
    install_instructions()
    
    try:
        # 1. Data Acquisition & Loading
        df_raw = load_landslide_data()
        
        # Filter for South Asia
        df_sa = filter_south_asia(df_raw)
        
        # 2. Data Cleaning & Preprocessing
        df_clean, X_cluster, X_class, y_class, preprocessor = clean_and_preprocess(df_sa)
        
        if len(df_clean) == 0:
            print("\n✗ No data available after preprocessing. Exiting.")
            sys.exit(1)
        
        # 3. Unsupervised Spatial Clustering
        clustering_results = perform_spatial_clustering(X_cluster, df_clean)
        if clustering_results:
            kmeans_final, cluster_labels, optimal_k, silhouette_scores = clustering_results
        
        # 4. Supervised Classification
        results, X_test, y_test = train_classification_models(X_class, y_class, preprocessor)
        
        # Plot ROC curves
        plot_roc_curves(results, y_test)
        
        # 5. Evaluation & Reporting
        evaluate_and_report(results, y_test, X_test)
        
        # Save processed data
        output_path = 'processed_landslide_data.csv'
        df_clean.to_csv(output_path, index=False)
        print(f"\n✓ Saved processed data to: {output_path}")
        
        # Execution summary
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n✓ Analysis completed successfully in {duration}")
        print("=" * 80)
        print("Generated files:")
        print("  - landslide_spatial_clusters.png (Spatial clustering visualization)")
        print("  - roc_curves.png (ROC curves for all models)")
        print("  - processed_landslide_data.csv (Cleaned processed data)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
