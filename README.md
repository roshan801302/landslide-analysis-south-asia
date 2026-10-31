<div align="center">

# 🌧️ Landslide Risk Analysis

### Mapping South Asian hotspots and exploring fatality patterns with machine learning

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python 3">
  <img src="https://img.shields.io/badge/Focus-South%20Asia-2E8B57" alt="South Asia">
  <img src="https://img.shields.io/badge/Methods-Clustering%20%7C%20Classification-6C63FF" alt="Clustering and classification">
  <img src="https://img.shields.io/badge/Status-Research%20Project-555" alt="Research project">
</p>

**An end-to-end data science project for exploring landslide events across the Himalayas and Western Ghats.**

[Overview](#-overview) · [Workflow](#-analysis-workflow) · [Visuals](#-project-visuals) · [Quick start](#-quick-start) · [Limitations](#-important-limitations)

</div>

---

## 🌍 Overview

This project analyzes historical landslide events in South Asia. It combines geographic filtering and data preparation with two machine learning tasks:

- **Spatial clustering** groups event locations to help explore geographic concentrations.
- **Fatality classification** compares models that distinguish events with reported fatalities from those without.

The analysis focuses on **India, Nepal, Pakistan, Bangladesh, Sri Lanka, and Bhutan**, with geographic bounds emphasizing the **Himalayas** and **Western Ghats**.

> **Use responsibly:** This is a research and educational analysis—not a validated operational early-warning or emergency-response system.

## 🖼️ Project visuals

Sample outputs from the analysis are shown below. **Select an image to view it at full size.**

<table>
  <tr>
    <td align="center" width="50%">
      <a href="landslide_spatial_clusters.png">
        <img src="landslide_spatial_clusters.png" width="100%" alt="K-means clustering of landslide event locations across the Western Ghats and Himalayas">
      </a>
    </td>
    <td align="center" width="50%">
      <a href="roc_curves.png">
        <img src="roc_curves.png" width="100%" alt="ROC curves comparing four fatality classification models">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Spatial clusters</strong><br>
      K-means groups event coordinates. Colors indicate cluster membership, not hazard severity.
    </td>
    <td align="center">
      <strong>Classifier comparison</strong><br>
      ROC curves compare fatality classifiers across probability thresholds; the dashed line is random ranking.
    </td>
  </tr>
</table>

<p align="center"><sub>These are example outputs already present in the project folder. Running the script regenerates the plots from the dataset it loads.</sub></p>

## ✨ What the project does

| Stage | Purpose |
| --- | --- |
| Load data | Read a local landslide CSV, or create synthetic sample data if it is unavailable |
| Filter events | Focus on selected South Asian countries, rainfall-related triggers, and study-area bounds |
| Prepare features | Handle selected missing values and derive month, year, season, and fatality class |
| Explore hotspots | Cluster latitude/longitude coordinates and compare candidate cluster counts |
| Compare classifiers | Train four models and report classification metrics |
| Save results | Export processed records and create visualizations |

## 🧭 Analysis workflow

### 1. Data loading

The script looks for `Global_Landslide_Catalog_Export_rows.csv` in its configured local and relative paths. If no candidate can be loaded, it generates a synthetic dataset with example event dates, locations, triggers, and fatality counts.

The synthetic data makes it possible to exercise the pipeline without the source CSV. **Synthetic results are demonstrations only and must not be interpreted as findings about real landslide risk.**

### 2. Geographic and trigger filtering

The analysis keeps records from the selected South Asian countries and looks for rainfall-related trigger descriptions. It then applies approximate bounding boxes:

| Study area | Latitude | Longitude |
| --- | ---: | ---: |
| Western Ghats | 8°–20° N | 72°–78° E |
| Himalayas | 25°–36° N | 70°–95° E |

These are broad filters for exploratory analysis, not precise administrative or geological boundaries. If the combined filters return no events, the script falls back to the available South Asian records.

### 3. Cleaning and feature engineering

The script drops rows without usable coordinates, fills missing fatality counts with zero, and marks missing trigger values as `unknown`. It creates a binary target:

- `fatal_class = 1` when the reported fatality count is greater than zero
- `fatal_class = 0` otherwise

Dates are used to derive month, year, and a broad season category (`winter`, `spring`, `monsoon`, or `post_monsoon`). For classification, numeric features are imputed and scaled; categorical features are imputed and one-hot encoded within a scikit-learn pipeline.

### 4. Spatial clustering

K-means is applied to event latitude and longitude. Candidate cluster counts are compared using the silhouette score, which measures how closely grouped points are relative to neighboring groups. A scatter plot shows event clusters and their centers.

The clusters are **descriptive groupings of recorded coordinates**, not validated hazard zones or predictions of future events.

### 5. Fatality classification

The script uses a stratified train/test split and compares:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (k-NN)
- Gaussian Naive Bayes

Reported metrics include accuracy, fatal-class precision, fatal-class recall, fatal-class F1-score, ROC-AUC, and a confusion matrix. Recall is highlighted because missed fatal events are important to examine; however, no model output should be used as a real-world warning without rigorous validation.

### 6. Evaluation and visualization

The pipeline prints model-level classification reports and a comparison table, then saves a spatial cluster visualization and ROC curves. It also exports the cleaned analysis data for further review.

## 🧮 Mathematical methods and calculations

This section gives a simple explanation of how the analysis makes groups and checks its predictions.

### 1. Fatal or non-fatal?

The script turns each event's fatality count into one of two labels:

```text
fatal_class = 1, if fatality_count > 0
fatal_class = 0, if fatality_count = 0
```

The model predicts the category, not the number of people affected.

**Example:** An event with 3 reported fatalities gets label `1`; an event with 0 gets label `0`.

### 2. Preparing the input data

Before training, the pipeline prepares the input features:

- Missing numeric values are filled with the median from the training data.
- Numeric features such as latitude, longitude, and month are standardized so their scales are more comparable.
- Text categories such as season and trigger are converted into indicator columns (one-hot encoding).

Standardization follows this idea:

```text
z = (x - mean) / standard deviation
```

The pipeline learns these preparation values from training data, then applies them to test data.

**Example:** If a training-set feature has an average of 10 and a standard deviation of 2, then for `x = 14`:

```text
z = (14 - 10) / 2 = 2
```

### 3. Grouping locations with K-means

K-means groups events by their latitude and longitude. It starts with cluster centers, assigns each event to its nearest center, and moves each center to the average location of its assigned events. It repeats this process to make the groups more compact.

The center of a cluster is the average of its event coordinates:

```text
center latitude  = sum of latitudes / number of events
center longitude = sum of longitudes / number of events
```

The script tries 2, 3, 4, 5, and 6 clusters. It chooses the group count with the best average silhouette score.

**Simple example:** If two events are at `(15, 75)` and `(16, 76)`, their average center is:

```text
latitude  = (15 + 16) / 2 = 15.5
longitude = (75 + 76) / 2 = 75.5
center    = (15.5, 75.5)
```

### 4. Checking cluster quality: silhouette score

The silhouette score checks whether events are closer to their own group than to other groups. It ranges from **-1 to 1**:

```text
silhouette = (distance to other group - distance to own group)
             / larger of those two distances
```

- Near **1**: groups are well separated.
- Near **0**: groups overlap.
- Below **0**: some events may fit better in another group.

This score measures how clearly the coordinates form groups. It does not measure landslide danger.

**Example:** A score of `0.72` suggests fairly distinct groups; a score of `0.05` suggests the groups overlap substantially.

> **Location note:** The script measures distances directly in latitude/longitude degrees. These are not exact ground distances, so treat the clusters as exploratory.

### 5. Training and testing the models

The data is split into:

- **80% training data** — used to teach the models.
- **20% test data** — used to check their predictions on records they did not train on.

The split is random but keeps roughly the same fatal/non-fatal proportions in each part. It is not a test of future events or entirely new locations.

**Example:** With 1,000 records, an 80/20 split gives about 800 training records and 200 test records.

The four models learn in different ways:

- **Logistic Regression** estimates the chance that an event is fatal.
- **Decision Tree** makes predictions through a sequence of feature-based decisions.
- **k-Nearest Neighbors** looks at the five most similar training examples and uses their labels.
- **Gaussian Naive Bayes** estimates class probabilities using a simplifying assumption that features are independent.

**Example:** If a classifier assigns an event a fatal probability of `0.80`, its default classification threshold of `0.50` labels it fatal. This probability is a model estimate, not a guarantee.

### 6. Understanding the prediction scores

For the fatal class, predictions can be grouped like this:

| | Model says non-fatal | Model says fatal |
| --- | ---: | ---: |
| **Actually non-fatal** | Correctly rejected | False alarm |
| **Actually fatal** | Missed fatal event | Correctly detected |

The script summarizes these results with:

- **Accuracy:** How often the model is correct overall.
- **Precision:** Of the events called fatal, how many were actually fatal?
- **Recall:** Of the actually fatal events, how many did the model find?
- **F1-score:** A combined score for precision and recall.

```text
accuracy  = (correct fatal + correct non-fatal) / all events
precision = correct fatal / (correct fatal + false alarms)
recall    = correct fatal / (correct fatal + missed fatal)
F1        = 2 * precision * recall / (precision + recall)
```

For example, suppose the model correctly detects **18** fatal events, misses **2**, raises **10** false alarms, and correctly identifies **70** non-fatal events:

```text
Accuracy  = (18 + 70) / 100       = 88%
Precision = 18 / (18 + 10)        ≈ 64%
Recall    = 18 / (18 + 2)         = 90%
F1        = 2 * 0.64 * 0.90 / (0.64 + 0.90) ≈ 75%
```

In this example, the model finds most fatal events (high recall), but some events it flags are false alarms (lower precision). These numbers are only an example, not results from this project.

### 7. ROC curve and ROC-AUC

The ROC curve compares how many fatal events a model detects against how many non-fatal events it incorrectly flags, as the probability cutoff changes. **ROC-AUC** summarizes this comparison in one number:

```text
true positive rate = correctly detected fatal / all actually fatal
false positive rate = false alarms / all actually non-fatal
```

- **1.0** means the model separates the two classes very well.
- **0.5** is about the same as random ranking.

ROC-AUC does not choose a cutoff for real-world decisions. It is a comparison measure, not proof that a model is safe or reliable.

**Example:** An ROC-AUC of `0.80` means that, for a randomly selected fatal and non-fatal event, the model ranks the fatal event higher about 80% of the time (subject to ties). It does not mean 80% of predictions are correct.

## 📊 Generated outputs

Running the script from the project directory creates or updates:

| Output | Description |
| --- | --- |
| `landslide_spatial_clusters.png` | Scatter plot of coordinate clusters and centers |
| `roc_curves.png` | ROC curves comparing classifier discrimination |
| `processed_landslide_data.csv` | Filtered and processed records, including derived fields |

Outputs are written relative to the current working directory.

The figures shown in [Project visuals](#-project-visuals) are the corresponding image files. They are tracked separately from the README, so retain those PNG files alongside this document for the images to display in a repository.

## 🚀 Quick start

### Requirements

- Python 3
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

### Install dependencies

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the analysis

From the project directory:

```bash
python run_landslide_analysis.py
```

To use the real-data path, place `Global_Landslide_Catalog_Export_rows.csv` in the project directory. Check the console output to confirm whether the CSV or synthetic fallback was used.

## 🗂️ Project files

```text
.
├── run_landslide_analysis.py          # Analysis pipeline
├── Global_Landslide_Catalog_Export_rows.csv  # Optional input dataset
├── LANDSLIDE_ANALYSIS_SUMMARY.md      # Detailed project documentation
├── processed_landslide_data.csv       # Generated processed data
├── landslide_spatial_clusters.png     # Generated clustering plot
└── roc_curves.png                     # Generated ROC plot
```

Generated files may not exist until the script has run.

## 🧰 Tools and methods

- **Data handling:** pandas, NumPy
- **Modeling:** scikit-learn pipelines, K-means, Logistic Regression, Decision Tree, k-NN, Gaussian Naive Bayes
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** silhouette score, precision, recall, F1-score, ROC-AUC, confusion matrix

## ⚠️ Important limitations

- A missing fatality count is treated as zero by the current script; in real datasets, missing may mean **unknown**, not **no fatalities**.
- The synthetic fallback is generated data and cannot support conclusions about actual events.
- Geographic bounds and trigger matching are approximate and may omit relevant events or include unrelated ones.
- The classifier uses a random stratified split, not a time-based or location-based validation. Performance may therefore overstate how well it generalizes to future events or new regions.
- Historical event records and available features may be incomplete, biased, or inconsistent.
- Clustering results depend on the input records and chosen method; they do not establish causal factors or forecast future landslides.
- This project has not been established as a safety-critical decision tool.

## 🌱 Research alignment

The project explores themes connected to:

- **UN SDG 11:** Sustainable Cities and Communities
- **UN SDG 15:** Life on Land

## 📚 More documentation

See [`LANDSLIDE_ANALYSIS_SUMMARY.md`](LANDSLIDE_ANALYSIS_SUMMARY.md) for a fuller explanation of the project design and analysis stages.

---

<div align="center">

**For research and educational exploration of landslide data in South Asia.**

</div>
