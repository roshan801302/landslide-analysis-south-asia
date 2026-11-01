#!/usr/bin/env python3
"""
Review-oriented landslide machine-learning survey and illustrative analysis.

The experiment compares fatality classifiers and coordinate-clustering methods
on South Asian records from the Global Landslide Catalog.

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
from matplotlib.colors import BoundaryNorm, ListedColormap

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

SOUTH_ASIA_COUNTRIES = (
    'Afghanistan',
    'Bangladesh',
    'Bhutan',
    'India',
    'Maldives',
    'Nepal',
    'Pakistan',
    'Sri Lanka',
)
SOUTH_ASIA_MAP_EXTENT = (60, 100, 0, 40)
SPATIAL_CLUSTER_COUNT = 3


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
    project_dir = Path(__file__).resolve().parent
    potential_paths = [
        Path(file_path) if file_path is not None else None,
        project_dir / "Global_Landslide_Catalog_Export_rows.csv",
        Path.cwd() / "Global_Landslide_Catalog_Export_rows.csv",
        project_dir / "data" / "Global_Landslide_Catalog_Export_rows.csv",
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
        print("This creates demonstration records across the South Asia analysis extent")
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
    
    # Approximate South Asian locations for pipeline demonstrations.
    
    data = []
    
    # Define regions
    regions = [
        # Western Ghats (India)
        {'country': 'India', 'lat_range': (8, 20), 'lon_range': (72, 78), 'weight': 0.32},
        # Himalayas - India
        {'country': 'India', 'lat_range': (25, 35), 'lon_range': (75, 95), 'weight': 0.24},
        # Himalayas - Nepal
        {'country': 'Nepal', 'lat_range': (26, 30), 'lon_range': (80, 88), 'weight': 0.15},
        # Himalayas - Pakistan
        {'country': 'Pakistan', 'lat_range': (30, 36), 'lon_range': (70, 77), 'weight': 0.08},
        # Bangladesh (hill tracts)
        {'country': 'Bangladesh', 'lat_range': (22, 26), 'lon_range': (91, 93), 'weight': 0.05},
        # Sri Lanka (highlands)
        {'country': 'Sri Lanka', 'lat_range': (6, 9), 'lon_range': (80, 81), 'weight': 0.05},
        # Bhutan
        {'country': 'Bhutan', 'lat_range': (27, 28), 'lon_range': (89, 92), 'weight': 0.06},
        # Afghanistan
        {'country': 'Afghanistan', 'lat_range': (29, 38), 'lon_range': (60, 75), 'weight': 0.05},
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
    Filter dataset to South Asian countries and rainfall-related events.
    
    Args:
        df (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Filtered South Asia dataset
    """
    print("\n" + "=" * 80)
    print("FILTERING FOR SOUTH ASIA")
    print("=" * 80)
    
    # Keep events from across South Asia rather than clipping to two subregions.
    if 'country_name' in df.columns:
        df_filtered = df[df['country_name'].isin(SOUTH_ASIA_COUNTRIES)].copy()
    else:
        if {'latitude', 'longitude'}.issubset(df.columns):
            min_lon, max_lon, min_lat, max_lat = SOUTH_ASIA_MAP_EXTENT
            in_extent = (
                df['latitude'].between(min_lat, max_lat)
                & df['longitude'].between(min_lon, max_lon)
            )
            df_filtered = df.loc[in_extent].copy()
        else:
            df_filtered = df.copy()
    
    print(f"Records after country filter: {len(df_filtered)}")

    if 'landslide_trigger' in df_filtered.columns:
        triggers = df_filtered['landslide_trigger'].astype('string').str.strip().str.lower()
        rainfall_triggers = {
            'rain',
            'downpour',
            'monsoon',
            'continuous_rain',
            'tropical_cyclone',
            'snowfall_snowmelt',
            'rainfall',
        }
        rainfall_mask = triggers.isin(rainfall_triggers) | triggers.str.contains(
            r'rain|monsoon|downpour|cyclone', na=False
        )
        rainfall_events = df_filtered.loc[rainfall_mask].copy()
        print(f"Records after rainfall trigger filter: {len(rainfall_events)}")

        if rainfall_events.empty and not df_filtered.empty:
            print(
                "\n⚠ No rainfall-triggered South Asian events found. "
                "Using all South Asian catalog events."
            )
        else:
            df_filtered = rainfall_events

    print(f"✓ Final filtered dataset: {len(df_filtered)} records")
    if 'country_name' in df_filtered.columns:
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
    
    # Keep unknown fatality labels missing; absence of a report is not evidence
    # that an event had zero fatalities.
    if 'fatality_count' in df_clean.columns:
        df_clean['fatality_count'] = pd.to_numeric(
            df_clean['fatality_count'], errors='coerce'
        )
        unknown_fatalities = int(df_clean['fatality_count'].isna().sum())
        print(f"  ✓ Kept {unknown_fatalities} missing fatality counts as unknown")
    
    # Fill missing trigger with 'unknown'
    if 'landslide_trigger' in df_clean.columns:
        df_clean['landslide_trigger'] = df_clean['landslide_trigger'].fillna('unknown')
        print(f"  ✓ Filled missing triggers with 'unknown'")
    
    # Create binary target column for classification: fatal_class = 1 if fatalities > 0, else 0
    print("\nCreating target variable...")
    known_fatality = df_clean['fatality_count'].notna()
    df_clean['fatal_class'] = (
        df_clean['fatality_count'].gt(0).astype(float).where(known_fatality)
    )
    class_dist = df_clean.loc[known_fatality, 'fatal_class'].value_counts()
    known_count = int(known_fatality.sum())
    print(f"  - Fatal (1): {class_dist.get(1, 0)}")
    print(f"  - Non-fatal (0): {class_dist.get(0, 0)}")
    print(f"  - Known fatality labels: {known_count} of {len(df_clean)} events")
    
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
    
    labeled_events = df_clean['fatal_class'].notna()
    X_class = df_clean.loc[labeled_events, feature_cols].copy()
    y_class = df_clean.loc[labeled_events, 'fatal_class'].astype(int).copy()
    
    print(f"  - Clustering features: {list(X_cluster.columns)}")
    print(f"  - Classification features: {list(X_class.columns)}")
    print(f"  - Labeled events used for classification: {len(X_class)}")
    
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


def perform_temporal_analysis(df_clean, forecast_horizon=12, save_plot=True):
    """Project future monthly catalog counts using historical calendar-month means."""
    print("\n" + "=" * 80)
    print("3. MONTHLY LANDSLIDE TRENDS & SEASONAL FORECAST")
    print("=" * 80)

    if 'event_date_parsed' not in df_clean.columns:
        print("⚠ Event dates are unavailable; skipping temporal analysis.")
        return None

    event_dates = df_clean['event_date_parsed'].dropna()
    if len(event_dates) == 0:
        print("⚠ No valid event dates; skipping temporal analysis.")
        return None

    monthly_counts = (
        df_clean.dropna(subset=['event_date_parsed'])
        .set_index('event_date_parsed')
        .resample('MS')
        .size()
    )
    monthly_counts = monthly_counts.reindex(
        pd.date_range(monthly_counts.index.min(), monthly_counts.index.max(), freq='MS'),
        fill_value=0,
    )

    if len(monthly_counts) < 24:
        print("⚠ At least 24 monthly observations are needed for a one-year holdout.")
        return None

    train = monthly_counts.iloc[:-12]
    test = monthly_counts.iloc[-12:]
    training_seasonal_means = train.groupby(train.index.month).mean()
    validation_forecast = pd.Series(
        [training_seasonal_means.get(month, train.mean()) for month in test.index.month],
        index=test.index,
    )
    mae = float(np.mean(np.abs(test.to_numpy() - validation_forecast.to_numpy())))
    rmse = float(np.sqrt(np.mean((test.to_numpy() - validation_forecast.to_numpy()) ** 2)))
    print(f"Recorded-event coverage: {monthly_counts.index.min():%Y-%m} to {monthly_counts.index.max():%Y-%m}")
    print(f"Chronological holdout: {test.index.min():%Y-%m} to {test.index.max():%Y-%m}")
    print(f"Seasonal-month baseline | MAE: {mae:.2f} events/month | RMSE: {rmse:.2f}")

    seasonal_means = monthly_counts.groupby(monthly_counts.index.month).mean()
    last_observed_month = monthly_counts.index.max()
    current_month = pd.Timestamp.now().to_period('M').to_timestamp()
    forecast_anchor = max(last_observed_month, current_month)
    forecast_dates = pd.date_range(
        forecast_anchor + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq='MS',
    )
    forecast = pd.DataFrame({
        'month': forecast_dates,
        'forecast_recorded_events': [
            float(seasonal_means.get(month, monthly_counts.mean()))
            for month in forecast_dates.month
        ],
        'latest_observed_month': last_observed_month,
        'forecast_basis': 'historical_calendar_month_mean',
    })
    forecast.to_csv('landslide_monthly_forecast.csv', index=False)
    print(
        f"✓ Saved future forecast for {forecast_dates.min():%Y-%m} to "
        f"{forecast_dates.max():%Y-%m} to landslide_monthly_forecast.csv"
    )
    months_since_observation = (
        current_month.to_period('M') - last_observed_month.to_period('M')
    ).n
    if months_since_observation > 12:
        print(
            f"⚠ Latest catalog month is {last_observed_month:%Y-%m} "
            f"({months_since_observation} months before the current month). "
            "Future values are historical seasonal averages, not a reliable "
            "current landslide-hazard forecast."
        )

    if save_plot:
        plt.figure(figsize=(13, 6))
        plt.plot(monthly_counts.index, monthly_counts.values, color='#5470c6',
                 alpha=0.65, label='Recorded events per month')
        plt.plot(monthly_counts.index, monthly_counts.rolling(12, min_periods=1).mean(),
                 color='#2f855a', linewidth=2, label='Trailing 12-month average')
        plt.plot(test.index, validation_forecast.values, color='#ed8936',
                 linestyle='--', linewidth=2, label='Holdout seasonal baseline')
        plt.plot(forecast['month'], forecast['forecast_recorded_events'],
                 color='#c53030', marker='o', linestyle='--',
                 label=f'Future {forecast_horizon} months: historical seasonal baseline')
        plt.axvline(last_observed_month, color='gray', linestyle=':', linewidth=1,
                    label='Latest observed catalog month')
        plt.title('Historical Monthly Records and Future Seasonal-Baseline Projection')
        plt.xlabel('Month')
        plt.ylabel('Recorded events')
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig('landslide_monthly_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved temporal analysis plot to landslide_monthly_forecast.png")

    return {
        'monthly_counts': monthly_counts,
        'validation_mae': mae,
        'validation_rmse': rmse,
        'forecast': forecast,
    }


def perform_seasonal_country_clustering(df_clean, min_events=12, save_plot=True):
    """Cluster countries by their normalized seasonal distribution of recorded events."""
    print("\n" + "=" * 80)
    print("4. SEASONAL CLUSTERING OF SOURCE COUNTRIES")
    print("=" * 80)

    required = {'country_name', 'season'}
    if not required.issubset(df_clean.columns):
        print("⚠ Country or season fields are unavailable; skipping seasonal clustering.")
        return None

    seasons = ['winter', 'spring', 'monsoon', 'post_monsoon']
    valid = df_clean.loc[
        df_clean['season'].isin(seasons) & df_clean['country_name'].notna()
    ].copy()
    valid['country_name'] = valid['country_name'].astype(str).str.strip()
    valid = valid[valid['country_name'].ne('')]
    counts = valid.groupby('country_name').size()
    eligible_countries = counts[counts >= min_events].index
    valid = valid[valid['country_name'].isin(eligible_countries)]

    if valid['country_name'].nunique() < 3:
        print(f"⚠ Fewer than three countries have at least {min_events} dated events; skipping clustering.")
        return None

    seasonal_counts = (
        valid.groupby(['country_name', 'season'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=seasons, fill_value=0)
    )
    seasonal_profiles = seasonal_counts.div(seasonal_counts.sum(axis=1), axis=0)
    unique_profiles = seasonal_profiles.drop_duplicates().shape[0]
    max_k = min(5, len(seasonal_profiles) - 1, unique_profiles)
    if max_k < 2:
        print("⚠ Country seasonal profiles do not contain enough distinct patterns to cluster.")
        return None

    silhouette_scores = {}
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(seasonal_profiles)
        silhouette_scores[k] = silhouette_score(seasonal_profiles, labels)
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    labels = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(seasonal_profiles)

    result = seasonal_profiles.mul(100).round(2)
    result.insert(0, 'event_count', seasonal_counts.sum(axis=1))
    result['seasonal_cluster'] = labels
    result.index.name = 'country_name'
    result = result.sort_values(['seasonal_cluster', 'country_name'])
    result.to_csv('country_seasonal_clusters.csv')

    print(f"Eligible countries: {len(result)} (minimum {min_events} dated events each)")
    print(f"Selected k={optimal_k} (silhouette score {silhouette_scores[optimal_k]:.3f})")
    print(result.to_string())
    print("✓ Saved seasonal country profiles to country_seasonal_clusters.csv")

    if save_plot:
        profile_columns = seasons
        plot_data = result[profile_columns]
        plt.figure(figsize=(10, max(4, 0.6 * len(plot_data))))
        sns.heatmap(
            plot_data,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Share of country events (%)'},
        )
        plt.title(f'Seasonal Distribution of Landslide Records by Country (k={optimal_k})')
        plt.xlabel('Season')
        plt.ylabel('Country')
        plt.tight_layout()
        plt.savefig('landslide_seasonal_country_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved seasonal clustering plot to landslide_seasonal_country_clusters.png")

    return result


def perform_spatial_clustering(
    X_cluster,
    df_clean,
    save_plot=True,
    n_clusters=SPATIAL_CLUSTER_COUNT,
):
    """
    Perform unsupervised spatial clustering for hotspot identification.
    
    Args:
        X_cluster (pd.DataFrame): Latitude/longitude coordinates
        df_clean (pd.DataFrame): Cleaned dataframe
        save_plot (bool): Whether to save the plot
    
    Returns:
        tuple: (kmeans_model, cluster_labels, cluster_count, silhouette_scores)
    """
    print("\n" + "=" * 80)
    print("5. UNSUPERVISED SPATIAL CLUSTERING (HOTSPOT IDENTIFICATION)")
    print("=" * 80)
    
    if len(X_cluster) < 2:
        print("⚠ Insufficient data for clustering")
        return None, np.array([]), 2, []

    unique_locations = X_cluster.drop_duplicates().shape[0]
    cluster_count = min(n_clusters, unique_locations)
    if cluster_count < 2:
        print("⚠ Insufficient unique locations for clustering")
        return None, np.array([]), cluster_count, []

    print(f"Clustering South Asian locations with K-means (k={cluster_count})...")
    kmeans_final = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_cluster)

    silhouette_scores = [silhouette_score(X_cluster, cluster_labels)]
    print(f"  Silhouette Score (k={cluster_count}): {silhouette_scores[0]:.4f}")

    print("\nPerforming Hierarchical Clustering (Agglomerative) for comparison...")
    hier = AgglomerativeClustering(n_clusters=cluster_count)
    hier_labels = hier.fit_predict(X_cluster)
    hier_sil = silhouette_score(X_cluster, hier_labels)
    print(f"  Hierarchical Clustering Silhouette Score (k={cluster_count}): {hier_sil:.4f}")
    
    # Add labels to dataframe
    df_clean['cluster_kmeans'] = cluster_labels
    df_clean['cluster_hierarchical'] = hier_labels
    
    # Use discrete colors and ticks because cluster IDs are categories, not values.
    if save_plot:
        print("\nGenerating spatial cluster visualization...")
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.get_cmap('tab10').colors[:cluster_count]
        cluster_cmap = ListedColormap(colors)
        cluster_norm = BoundaryNorm(
            np.arange(-0.5, cluster_count + 0.5, 1),
            cluster_cmap.N,
        )
        points = ax.scatter(
            X_cluster['longitude'],
            X_cluster['latitude'],
            c=cluster_labels,
            cmap=cluster_cmap,
            norm=cluster_norm,
            s=34,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.25,
        )
        colorbar = fig.colorbar(
            points,
            ax=ax,
            ticks=np.arange(cluster_count),
            boundaries=np.arange(-0.5, cluster_count + 0.5, 1),
            spacing='uniform',
        )
        colorbar.set_ticklabels(
            [f'Cluster {cluster_id + 1}' for cluster_id in range(cluster_count)]
        )
        colorbar.set_label('Cluster')
        min_lon, max_lon, min_lat, max_lat = SOUTH_ASIA_MAP_EXTENT
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_aspect(1 / np.cos(np.deg2rad(20)))
        annotation_style = {
            'fontsize': 9,
            'style': 'italic',
            'bbox': {
                'boxstyle': 'round,pad=0.25',
                'facecolor': 'white',
                'alpha': 0.8,
            },
        }
        ax.text(72.5, 15, 'Western Ghats', **annotation_style)
        ax.text(82, 30, 'Himalayas', **annotation_style)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(
            f'Spatial Clustering of Landslide Hotspots (K-means, k={cluster_count})\n'
            'South Asia',
            fontsize=14,
            fontweight='bold',
        )
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = 'landslide_spatial_clusters.png'
        fig.savefig(plot_path, dpi=240, bbox_inches='tight')
        print(f"✓ Saved cluster plot to: {plot_path}")
        plt.close(fig)
    
    # Cluster statistics
    print("\nCluster Statistics (K-means):")
    for i in range(cluster_count):
        cluster_mask = cluster_labels == i
        n_points = cluster_mask.sum()
        mean_lat = X_cluster[cluster_mask]['latitude'].mean()
        mean_lon = X_cluster[cluster_mask]['longitude'].mean()
        fatalities = df_clean[cluster_mask]['fatality_count'].sum()
        fatal_events = df_clean[cluster_mask]['fatal_class'].sum()
        print(f"  Cluster {i}: {n_points} events | Center: ({mean_lat:.2f}, {mean_lon:.2f}) | Total fatalities: {fatalities} | Fatal events: {fatal_events}")
    
    return kmeans_final, cluster_labels, cluster_count, silhouette_scores


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
    print("6. SUPERVISED CLASSIFICATION (FATAL VS. NON-FATAL)")
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
    print("7. EVALUATION & REPORTING")
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
            'Accuracy': res['accuracy'],
            'Precision (Fatal)': res['precision'],
            'Recall (Fatal)': res['recall'],
            'F1-Score (Fatal)': res['f1'],
            'ROC-AUC': res['roc_auc'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('classification_metrics.csv', index=False)
    print(comparison_df.to_string(
        index=False,
        formatters={
            column: '{:.4f}'.format
            for column in comparison_df.columns
            if column != 'Model'
        },
    ))
    print("✓ Saved classifier metrics to classification_metrics.csv")
    
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
    print("South Asia - illustrative fatality classification and coordinate clustering")
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
        
        # 3–4. Temporal trend forecasting and seasonal country clustering
        perform_temporal_analysis(df_clean)
        perform_seasonal_country_clustering(df_clean)

        # 5. Unsupervised Spatial Clustering
        perform_spatial_clustering(X_cluster, df_clean)
        
        # 6. Supervised Classification
        results, X_test, y_test = train_classification_models(X_class, y_class, preprocessor)
        
        # Plot ROC curves
        plot_roc_curves(results, y_test)
        
        # 7. Evaluation & Reporting
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
        print("  - landslide_monthly_forecast.png (Monthly counts and seasonal baseline)")
        print("  - landslide_monthly_forecast.csv (12-month exploratory forecast)")
        print("  - landslide_seasonal_country_clusters.png (Country seasonal profiles)")
        print("  - country_seasonal_clusters.csv (Country seasonal cluster assignments)")
        print("  - classification_metrics.csv (Fatality classifier comparison)")
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
