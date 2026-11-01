# A Focused Survey and South Asian Illustrative Experiment

## Abstract

Machine learning (ML) is widely used to analyze landslide inventories, terrain and environmental predictors, and event reports. The term “landslide prediction” can refer to distinct tasks: susceptibility mapping estimates relative spatial propensity, hazard analysis also considers occurrence time or magnitude, risk analysis incorporates exposure and vulnerability, and event classification assigns labels to already-recorded events. This focused narrative survey summarizes common ML approaches, validation concerns, and evaluation measures for landslide research. It accompanies an illustrative experiment on South Asian records from the Global Landslide Catalog (GLC): four classifiers distinguish recorded events with fatalities from events with an explicitly reported zero fatality count, while K-means and agglomerative clustering group event coordinates. In the fixed stratified split, the decision tree has the highest fatal-class recall and F1-score, while k-nearest neighbors (k-NN) has the highest ROC-AUC. Three-cluster K-means has a silhouette score of 0.607, compared with 0.573 for hierarchical clustering. These results demonstrate an analysis workflow, not an operational hazard model. The project’s global source file ends in 2017, its filtered South Asia subset ends in October 2016, and the 345 records with unknown fatality counts are excluded from classification.

**Keywords:** landslides, machine learning, susceptibility, event classification, clustering, South Asia, Global Landslide Catalog

## 1. Scope and review approach

This is a **focused narrative survey**, not a systematic review or meta-analysis. It synthesizes established landslide-susceptibility reviews, an overview of ML algorithm comparisons, foundational papers on ROC analysis and cluster validation, and the catalog paper associated with the experiment’s source data [1–5]. It does not claim exhaustive database searches, formal screening, or pooled performance estimates. The illustrative experiment connects the method discussion to a reproducible analysis; it is not evidence that one algorithm is generally superior.

The distinction between prediction targets is central. A model trained only on recorded landslide points cannot by itself learn the difference between landslide and stable terrain. Nor does a model that classifies fatalities among known events estimate where or when a new landslide will occur. The experiment below addresses only the narrower **event-outcome classification** question.

## 2. Problem framing and common data

### 2.1 Susceptibility, hazard, risk, and event outcomes

- **Susceptibility** describes where landslides are more likely relative to other locations, usually conditional on terrain and environmental factors.
- **Hazard** additionally addresses the probability of occurrence within a time period and, depending on the study, landslide magnitude or intensity.
- **Risk** combines hazard with exposed people or assets and their vulnerability.
- **Event classification** predicts an attribute of an event already in an inventory, such as a reported fatality label.

These are not interchangeable. In particular, a high spatial-cluster density is not a calibrated probability of slope failure, and a fatal/non-fatal classifier is not a susceptibility map. Reichenbach et al. review statistically based susceptibility models and emphasize the range of modeling choices in spatial assessment [1]. Merghadi et al. compare ML methods used in susceptibility studies and provide broader context for algorithm selection [2].

### 2.2 Inventories and predictors

Landslide inventories commonly combine event locations with terrain, geology, land cover, hydrology, and triggering conditions such as precipitation or earthquakes. Catalogs are valuable but are not complete censuses: reporting varies by place, time, event size, access, and information source. The GLC paper describes its compilation approach and limitations [3]. Inventory incompleteness can affect both the apparent spatial pattern and the labels a model is asked to learn.

For susceptibility mapping, predictor rasters must be aligned in spatial resolution, projection, extent, and time reference. Event points may need to be related to slope units or raster cells; the choice of spatial unit affects the samples and interpretation. Presence-only records also require a defensible comparison/background sample if the intended model is to distinguish landslide from non-landslide locations. Randomly drawing unverified “absence” points can introduce label noise.

## 3. Machine-learning method families

Landslide studies draw on multiple model families, not just the four used in this experiment.

| Model family | Typical examples | Common role | Main considerations |
|---|---|---|---|
| Statistical and linear | Logistic regression, generalized linear/additive models | Interpretable susceptibility baselines and event classification | Functional form and predictor dependence can limit fit; calibration and sampling assumptions still matter |
| Tree and ensemble | Decision trees, random forests, gradient-boosted trees | Tabular terrain-factor susceptibility or event classification | Trees capture nonlinearities; ensembles can improve stability but need tuning and careful validation |
| Kernel and distance-based | Support vector machines, k-NN | Classification from engineered spatial/environmental features | Scaling, kernel/distance choice, class imbalance, and sample density matter |
| Neural and deep learning | Multilayer networks, convolutional or sequence models | Large tabular, raster, image, or time-series inputs | Require adequate data, careful spatial/temporal splits, and interpretability checks |
| Unsupervised | K-means, hierarchical clustering, density-based clustering | Explore inventory groupings or spatial patterns without outcome labels | Clusters depend on metric, scale, density, and sampling; they are not hazard probabilities |

Comparative reviews emphasize that no single algorithm is best independently of inventory, predictors, sampling, study extent, and validation design [1, 2]. The illustrative implementation intentionally uses standard baselines rather than an exhaustive benchmark.

### 3.1 Supervised classifiers in this experiment

The experiment compares four familiar classifiers using latitude, longitude, month, season, and trigger category:

1. **Logistic regression** estimates a class probability through a linear decision function and is a useful interpretable baseline.
2. **Decision tree** recursively partitions the feature space into rules. It can express nonlinear interactions but may be unstable without regularization.
3. **k-nearest neighbors (k-NN)** classifies an example using nearby training examples. Distance scaling and the choice of \(k\) matter, especially for mixed or high-dimensional inputs.
4. **Gaussian Naive Bayes** combines class-conditional feature likelihoods under a conditional-independence assumption. The assumption is often approximate, but the model is inexpensive and provides a useful baseline.

Numeric imputation/scaling and categorical imputation/one-hot encoding are fit inside a scikit-learn pipeline, so transformations are learned from the training partition rather than the held-out test data.

### 3.2 Unsupervised coordinate clustering

**K-means** assigns points to \(k\) groups by minimizing within-cluster squared Euclidean distance to cluster centers. It is computationally efficient, but favors compact groups and depends on feature scaling and distance representation. **Agglomerative hierarchical clustering** repeatedly merges groups according to a linkage rule; this experiment uses Ward linkage. A dendrogram can make hierarchical results easier to inspect, though the experiment reports labels and a silhouette score rather than a dendrogram.

Both clustering methods here use only event latitude and longitude. Raw latitude/longitude degrees are not uniform ground-distance units: a degree of longitude changes physical length with latitude. The clusters should therefore be read as exploratory geographic groupings, not a physical-distance analysis or hazard-zone delineation.

## 4. Evaluation and validation

For the positive (fatal-event) class:

- **Recall** is the fraction of fatal events detected among all fatal events.
- **F1-score** is the harmonic mean of precision and recall at a chosen classification threshold.
- **ROC-AUC** measures ranking discrimination across thresholds; it does not select an operating threshold or establish probability calibration.

Accuracy alone can conceal poor detection when classes are imbalanced. Fawcett provides an accessible account of ROC curves and their interpretation [4]. For clustering, the **silhouette coefficient** compares within-cluster cohesion with separation from the nearest alternative cluster; scores near 1 indicate clearer separation, values near 0 indicate overlap, and negative values suggest some points may be assigned poorly [5]. Silhouette is not a measure of landslide danger or geographic completeness.

Validation design is as important as model choice. A random event-level split may put nearby events or events from the same reporting period in both training and test sets, yielding optimistic estimates for transfer to a new place or future time. Stronger claims require spatially blocked validation, temporal holdouts, or ideally external inventories, alongside checks for class imbalance, label uncertainty, calibration, and sensitivity to inventory quality.

## 5. Illustrative South Asian experiment

### 5.1 Data and protocol

The experiment uses `Global_Landslide_Catalog_Export_rows.csv`, a local export of the GLC [3]. The file contains **11,033 records**, with event dates from **November 7, 1988 through September 28, 2017**. The project selects rainfall-related records from Afghanistan, Bangladesh, Bhutan, India, Nepal, Pakistan, and Sri Lanka, then removes events without usable coordinates. The filtered inventory contains **1,845 events**, and its latest event is from **October 2016**. Maldives is within the project’s South Asia map viewport but has no records in this filtered subset.

There are **no observed 2018–2026 event records in the local source file**. Although the NASA Data.gov listing was updated as a page/dataset record in 2025, its description identifies the export as a one-time snapshot. A metadata update does not establish that the event rows have been updated. Recent-year statements must therefore not be inferred from this file.

Fatality counts are converted to a binary event label: a reported count greater than zero is fatal, and an explicitly reported zero is non-fatal. The **345 records with missing fatality counts remain unknown and are excluded from supervised classification**, rather than being treated as non-fatal. This leaves 1,500 labeled events: 601 fatal and 899 non-fatal. The classifier uses a stratified 80/20 random split with `random_state=42` (1,200 training and 300 test events; 120 fatal test events).

Coordinate clustering uses all 1,845 filtered records. K-means is fixed at \(k=3\) for the South Asia-wide visualization; it is not selected by searching for the largest silhouette score. Agglomerative clustering is evaluated at the same \(k\). Both use unprojected latitude/longitude coordinates.

### 5.2 Classifier results

| Classifier | Accuracy | Fatal precision | Fatal recall | Fatal F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.6267 | 0.5377 | 0.4750 | 0.5044 | 0.6094 |
| Decision tree | 0.6200 | 0.5246 | **0.5333** | **0.5289** | 0.6413 |
| k-NN | **0.6533** | 0.5833 | 0.4667 | 0.5185 | **0.6775** |
| Gaussian Naive Bayes | 0.6200 | 0.6000 | 0.1500 | 0.2400 | 0.6207 |

For this one split, the decision tree attains the highest fatal-class recall and F1-score, while k-NN attains the highest ROC-AUC and accuracy. The modest ROC-AUC values and lower recall for some models indicate limited discrimination in this setup. These model rankings are split-dependent and should not be generalized beyond this experiment.

### 5.3 Coordinate-clustering results

| Method | Clusters | Silhouette |
|---|---:|---:|
| K-means | 3 | 0.6070 |
| Agglomerative (Ward) | 3 | 0.5728 |

The K-means score is higher on this coordinate sample under the chosen representation. It does not mean K-means has identified validated hazard zones; it only indicates better separation under this particular silhouette calculation.

### 5.4 Monthly event-count baseline

The analysis also aggregates filtered catalog records by month, fills gaps between the first and last observed months with zero counts, and evaluates a calendar-month seasonal-average baseline on the final 12 observed months. For the current filtered data, the holdout runs from **November 2015 through October 2016**, with **MAE 7.87** and **RMSE 13.06 events per month**.

For the 12-month projection, the implementation uses historical average counts for the corresponding calendar months. Because the filtered data stops in October 2016, future-dated outputs such as December 2026–November 2027 are **not observed landslide data** and should not be interpreted as a current forecast. They are extrapolations from stale and incomplete records; recent reporting coverage is unknown. A useful present-day forecast needs a verified up-to-date event series, explicit missing-reporting treatment, suitable covariates, and evaluation on genuinely held-out recent periods.

## 6. Interpretation, limitations, and next steps

This exercise demonstrates how common classification metrics, clustering diagnostics, and a simple time-series baseline can be reported alongside an event inventory. It does **not** make a landslide occurrence forecast, susceptibility surface, warning system, or risk estimate. Important limitations are:

1. **Stale and incomplete inventory:** the local source ends in 2017 globally and the filtered South Asia series ends in 2016. Unrecorded events are not stable locations, and catalog counts reflect observation and reporting as well as occurrence.
2. **Narrow classification target:** fatality classification is conditional on an event being recorded and on its fatality count being known. Missing counts are excluded, but a recorded zero may still have reporting uncertainty.
3. **Weakly contextualized features:** coordinates, month, season, and trigger do not represent all relevant terrain, rainfall intensity, geology, exposure, or vulnerability.
4. **Random validation:** the single stratified split does not establish generalization to unseen regions or later years. Repeated spatial and temporal validation is needed before comparative claims.
5. **Distance and clustering assumptions:** latitude/longitude degrees are not equal-distance projected coordinates; cluster boundaries depend on \(k\), metric, sampling, and reporting density.
6. **Threshold and probability interpretation:** the default classifier threshold is not selected for a real response context, and ROC-AUC does not prove calibration or operational utility.
7. **Monthly zeros and forecast assumptions:** zero catalog entries may represent missing reporting rather than no events. Historical seasonal averages do not account for changes in reporting, exposure, climate, or triggering conditions.

Further work should use complete, documented inventories; retain unknown outcomes as unknown; add physically motivated covariates; compare spatial-block and temporal validation against the random-split baseline; test calibration and class-specific costs; and validate on independent regions or time periods. A susceptibility study would require an explicit spatial sampling design and a target defined for locations, rather than reusing this event-fatality label.

## References

1. Reichenbach, P., Rossi, M., Malamud, B. D., Mihir, M., & Guzzetti, F. (2018). A review of statistically-based landslide susceptibility models. *Earth-Science Reviews, 180*, 60–91. https://doi.org/10.1016/j.earscirev.2018.03.001
2. Merghadi, A., Yunus, A. P., Dou, J., Whiteley, J., Pham, B. T., Bui, D. T., Avtar, R., & Abderrahmane, B. (2020). Machine learning methods for landslide susceptibility studies: A comparative overview of algorithm performance. *Earth-Science Reviews, 207*, 103225. https://doi.org/10.1016/j.earscirev.2020.103225
3. Kirschbaum, D. B., Adler, R., Hong, Y., Hill, S., & Lerner-Lam, A. (2010). A global landslide catalog for hazard applications: Method, results, and limitations. *Natural Hazards, 52*(3), 561–575. https://doi.org/10.1007/s11069-009-9401-4
4. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
5. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

### Data source

NASA, [Global Landslide Catalog Export](https://catalog.data.gov/dataset/global-landslide-catalog-export). The local copy used for this analysis is `Global_Landslide_Catalog_Export_rows.csv`; consult the local date range above when reproducing or interpreting the results.