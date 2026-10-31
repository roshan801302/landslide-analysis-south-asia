# Landslide Analysis Project — Detailed Documentation

## 1. Overview

This project is a research-oriented machine learning workflow for analyzing landslide events in South Asia, with emphasis on the Himalayan belt and the Western Ghats. The main objective is to understand where landslide hotspots occur geographically and whether an event is likely to be fatal or non-fatal based on available attributes.

The script combines:

- data loading and validation
- geographic filtering
- data cleaning and feature engineering
- unsupervised clustering for hotspot detection
- supervised classification for fatality prediction
- visualization and report generation

The overall workflow is implemented in the file `run_landslide_analysis.py`.

---

## 2. Project Objective

The project addresses two core questions:

1. Where are the major landslide hotspot regions in South Asia?
2. Which landslide events are likely to be fatal and therefore require more attention for risk reduction and early warning?

This connects directly to public safety and disaster-risk analysis. The script also aligns with sustainable development goals:

- SDG 11: Sustainable Cities and Communities
- SDG 15: Life on Land

---

## 3. Data Sources and Input Strategy

### 3.1 Primary Data Source

The analysis attempts to use the NASA Global Landslide Catalog (GLC) dataset. The script checks for a local CSV file named:

- `Global_Landslide_Catalog_Export_rows.csv`

It looks for this file in common local paths and relative directories.

### 3.2 Fallback Strategy

If the real dataset is missing or cannot be loaded, the project creates a realistic synthetic dataset. This is specifically designed to resemble conditions in South Asia, especially:

- the Himalayas
- the Western Ghats
- India, Nepal, Pakistan, Bangladesh, Sri Lanka, and Bhutan

The synthetic generation is useful because it allows the full pipeline to run even when the original data source is unavailable.

### 3.3 Synthetic Data Characteristics

The fallback dataset is created with:

- latitude and longitude within South Asian geographic ranges
- realistic trigger values such as rain, monsoon, downpour, and cyclone-like conditions
- event date values between 2000 and 2020
- fatality counts skewed toward non-fatal events but with meaningful fatal cases
- country labels and event metadata similar to those used by real disaster datasets

This ensures the machine learning workflow remains executable and meaningful in testing or academic settings.

---

## 4. Data Loading and Initial Setup

The script defines a function called `load_landslide_data(file_path=None)`.

This function:

- checks possible file locations
- tries to read the CSV using pandas
- prints metadata about the dataset shape and columns
- falls back to synthetic data if no dataset is found

During load, it prints a summary including:

- number of rows
- number of columns
- first few column names
- date range if available

This gives a quick verification that the data is in the expected format before any further analysis.

---

## 5. Spatial Filtering for South Asia

The function `filter_south_asia(df)` narrows the dataset to relevant geography.

### 5.1 Country-Level Filtering

It keeps records from the following countries:

- India
- Nepal
- Pakistan
- Bangladesh
- Sri Lanka
- Bhutan

### 5.2 Trigger-Based Filtering

The script focuses on rainfall-triggered landslides by checking whether the `landslide_trigger` field contains terms such as:

- rain
- rainstorm
- monsoon
- downpour
- cyclone
- continuous rain
- snowfall/snowmelt

This makes the study more relevant to climate-driven landslides, which are common in the South Asian monsoon belt.

### 5.3 Geographic Bounding Filters

The script focuses on two important spatial regions:

- Western Ghats: approximate latitude 8°–20°N and longitude 72°–78°E
- Himalayas: approximate latitude 25°–36°N and longitude 70°–95°E

This filtered area reduces noise and focuses on the main physical zones where landslide hazard is significant.

### 5.4 Fallback if Filtering Produces No Data

If no records remain after applying those filters, the script falls back to the South Asian subset of the original data rather than failing completely.

---

## 6. Data Cleaning and Preprocessing

The function `clean_and_preprocess(df)` performs the main preprocessing pipeline.

### 6.1 Missing Value Handling

The script checks key attributes such as:

- latitude
- longitude
- event_date
- landslide_trigger
- fatality_count

Rows with missing coordinates are dropped because location is essential for spatial analysis.

Missing fatality counts are replaced with zero, since missing fatality information is treated as no reported fatalities by default.

Missing trigger values are replaced with `unknown` so the model can still process these rows.

### 6.2 Binary Target Creation

A target variable named `fatal_class` is created:

- `1` if fatality_count > 0
- `0` otherwise

This transforms the problem into a binary classification task: fatal vs. non-fatal landslide events.

### 6.3 Temporal Feature Engineering

The script extracts time-based features from `event_date`:

- month
- year
- season

The season logic is based on a Northern Hemisphere pattern:

- winter: December, January, February
- spring: March, April, May
- monsoon: June, July, August, September
- post_monsoon: October, November

This is important because South Asia experiences strong monsoon-driven landslide patterns.

### 6.4 Feature Separation for Different Tasks

The script builds two different feature sets:

1. `X_cluster` for clustering
   - latitude
   - longitude

2. `X_class` for classification
   - latitude
   - longitude
   - month
   - season
   - landslide_trigger

This division is important because clustering is purely spatial, while classification uses both geographic and environmental context.

### 6.5 ML Preprocessing Pipeline

For classification, the script uses a scikit-learn `ColumnTransformer` with separate preprocessing for numeric and categorical variables.

Numeric features include:

- latitude
- longitude
- month

These are processed using:

- median imputation
- standard scaling

Categorical features include:

- season
- landslide_trigger

These are processed using:

- constant value imputation for missing values
- one-hot encoding

This prevents data leakage and makes the pipeline consistent for model training.

---

## 7. Spatial Clustering of Hotspots

The function `perform_spatial_clustering(X_cluster, df_clean, save_plot=True)` analyzes spatial patterns in the dataset.

### 7.1 Why Clustering Is Used

The script wants to identify geographic clusters of high-risk zones. K-means is used as an unsupervised technique to group similar coordinate points into clusters.

### 7.2 K-means Configuration

The algorithm evaluates a range of cluster counts and computes silhouette scores to choose the best value of `k`.

Silhouette score helps answer:

- Are the clusters well separated?
- Are points in the same cluster actually similar?

This makes the clustering step more principled instead of arbitrarily choosing a fixed number of clusters.

### 7.3 Plotting the Result

The script visualizes the clusters using a scatter plot with:

- latitude on the x-axis
- longitude on the y-axis
- different colors for each cluster
- cluster centers marked explicitly

The output is saved as:

- `landslide_spatial_clusters.png`

### 7.4 Cluster Statistics Report

For each cluster, the script prints:

- number of events in the cluster
- mean latitude and longitude
- total fatalities
- number of fatal events

This helps interpret which geographic regions are most severe or high-risk.

---

## 8. Supervised Classification for Fatality Prediction

The function `train_classification_models(X, y, preprocessor, ...)` trains multiple classifiers to predict whether a landslide event is fatal or non-fatal.

### 8.1 Train-Test Split

The data is split into training and testing sets using an 80/20 ratio with `train_test_split` and `stratify=y` to keep the class distribution balanced.

The script prints the sample counts and training distribution.

### 8.2 Models Used

It trains four models:

- Logistic Regression
- Decision Tree Classifier
- k-Nearest Neighbors (k-NN)
- Gaussian Naive Bayes

The class weight is balanced for Logistic Regression and Decision Tree to account for class imbalance, since fatal landslide events are typically fewer than non-fatal ones.

### 8.3 Pipeline Structure

Each model is wrapped in a scikit-learn `Pipeline`:

- preprocessor
- classifier

This ensures the same transformation steps are applied consistently to training and test data.

### 8.4 Evaluation Metrics

For each model, the script computes:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix

The project emphasizes recall because missing a fatal landslide is more dangerous than a false alarm. In other words, for disaster risk modeling, correctly identifying fatal events is the most critical objective.

---

## 9. ROC Curve Visualization

The function `plot_roc_curves(results, y_test, save_plot=True)` creates ROC curves for all models.

It plots:

- true positive rate vs false positive rate
- a diagonal random baseline
- AUC labels on the curves

The resulting image is saved as:

- `roc_curves.png`

This helps compare overall classifier performance and choose the best model based on discrimination ability.

---

## 10. Evaluation and Reporting

The function `evaluate_and_report(results, y_test, X_test)` prints a detailed summary.

### 10.1 Classification Reports

For each model, the function prints:

- precision
- recall
- F1-score
- support
- confusion matrix

This provides a detailed understanding of how each model behaves across fatal and non-fatal classes.

### 10.2 Model Comparison Table

The script creates a summary table with the following fields:

- Model
- Accuracy
- Precision (Fatal)
- Recall (Fatal)
- F1-Score (Fatal)
- ROC-AUC

This is formatted so it can be copied into a research paper or report.

### 10.3 Key Findings Logic

The project determines:

- the best model by recall (for safety-critical screening)
- the best model by F1-score (balanced performance)

This is useful because different tasks may prioritize different metrics. For example:

- public safety may prioritize recall for fatal events
- balanced performance may prioritize F1-score

---

## 11. Output Files Generated by the Script

When the project runs successfully, it generates the following files:

- `landslide_spatial_clusters.png`
  - clustered spatial hotspot map

- `roc_curves.png`
  - ROC curves for all classification models

- `processed_landslide_data.csv`
  - cleaned and processed version of the filtered dataset

The script also prints summary tables to the console during execution.

---

## 12. Execution Flow of the Main Program

The `main()` function follows this sequence:

1. Print banner and execution details
2. Show installation instructions
3. Load the landslide dataset
4. Filter it to South Asia
5. Clean and preprocess it
6. Run spatial clustering
7. Train classification models
8. Plot ROC curves
9. Evaluate and summarize results
10. Save processed data
11. Display a completion summary

The script is designed to run as a complete analysis pipeline from start to finish.

---

## 13. How to Run the Project

Open a terminal in the project directory and execute:

```bash
python run_landslide_analysis.py
```

### Required Libraries

The script requires:

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Installation Command

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 14. Strengths of the Project

This project is useful because it combines both unsupervised and supervised learning:

- clustering reveals spatial structure in landslide activity
- classification aims to identify dangerous events before they escalate
- the workflow is easy to extend for more advanced models or real-world disaster data
- synthetic fallback ensures reproducibility even without a complete dataset

---

## 15. Limitations and Practical Considerations

Although the workflow is solid for academic and research use, some limitations should be noted:

- the real NASA dataset may vary in quality and missingness across event fields
- the synthetic fallback is not a replacement for actual historical records
- fatality prediction is constrained by the data features available in the dataset
- cluster detection depends on the selected geographic bounds and trigger filters
- the model performance is only as good as the data quality and quantity

---

## 16. Research Interpretation

In practical terms, the project can be interpreted as a prototype disaster-risk analytics system. It helps answer:

- where landslides are clustered in South Asia
- which regions might require more monitoring
- whether a landslide is likely to be fatal based on patterns in location, season, and trigger type

This is highly relevant for disaster preparedness, infrastructure planning, and early warning systems.

---

## 17. Summary

This project provides an end-to-end landslide analysis pipeline for South Asia using machine learning. It merges geographic analysis with predictive modeling to create a practical framework for exploring hazard patterns and fatality risk.

It is particularly useful for:

- academic research
- geospatial analysis
- machine learning education
- climatology and disaster-risk studies
- environmental risk assessment

In short, it demonstrates how data science can be used to turn a real-world hazard dataset into actionable spatial and predictive insights.
