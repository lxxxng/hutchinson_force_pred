﻿# 📊 Force Prediction from Stress Sensor Data

## 🎯 Objective
To develop a robust machine learning model that can predict **multi-directional force (X, Y, Z)** using stress sensor readings, enabling **indirect force estimation** in real-world applications where **direct measurement is impractical or impossible**.

---

## 🗂️ Repository Structure
```python
.
├── functions/
│ ├── init.py
│ └── feature_generation.py
│
├── notebooks/
│ ├── 01_load_data.ipynb # Data exploration, cleaning, inner join on time
│ ├── 02_visualization.ipynb # Exploratory visualization
│ ├── 03_remove_outliers.ipynb # IQR, Chauvenet, LOF, Isolation Forest
│ ├── 04_feature_engineering.ipynb # Feature generation, PCA, scaling, merging
│ ├── 05_model_training.ipynb # Feature selection and model training (RF, HGB, MLP)
│ └── 06_predict_test_data.ipynb # Preprocess test data, generate predictions
│
├── results/
│ └── predicted_forces.xlsx # Final prediction output
│
└── README.md
```

## ✅ Execution Flow

1. **01_load_data.ipynb**  
   Data loading, cleaning, and time-based joining.

2. **02_visualization.ipynb**  
   Exploratory visual analysis.

3. **03_remove_outliers.ipynb**  
   Outlier detection using **IQR, Chauvenet, LOF, Isolation Forest**.

4. **04_feature_engineering.ipynb**  
   Feature generation, PCA, scaling, and combining datasets.

---

## 🧠 Feature Engineering Overview

| **Feature Group**                | **Purpose**                                                               | **Benefit to Model**                                           |
|----------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------|
| Global Statistics (mean, std, sum)| Capture overall signal energy and variability across all sensors.         | Understand global system state or load intensity.              |
| Color-Based Statistics           | Capture color-specific dynamics (Red, Blue, Yellow).                      | Differentiate sensor behavior by color.                        |
| Rolling Window Features          | Capture short-term trends and local variability over time.                | Improve temporal awareness.                                   |
| Temporal Changes (diff)          | Highlight rate of change or dynamic shifts.                              | Detect spikes, shifts, or transitions.                         |
| Sensor Block Aggregations        | Summarize grouped sensor behavior (blocks 1-3, 4-6).                      | Reduce dimensionality while preserving spatial structure.      |
| Fourier-Based Features (optional)| Capture frequency-domain characteristics (cycles, periodic patterns).     | Detect cyclic or periodic behavior.                            |


5. **05_model_training.ipynb**  
   Feature selection and model training with **Random Forest, HGB, and MLP**.

---

## 🧪 Feature Set Strategies

| **Feature Set**                       | **Purpose**                                                                                     |
|--------------------------------------|-------------------------------------------------------------------------------------------------|
| Original Features                    | Baseline performance check with raw data.                                                       |
| PCA Features                         | Reduce dimensionality while preserving variance.                                                |
| Original + Rolling + Temporal        | Capture local trends and dynamic changes.                                                       |
| Original + Color + Block Aggregations| Understand sensor structure and grouped behavior.                                               |
| All Features                         | Evaluate combined perspectives at risk of redundancy.                                           |
| De-correlated Features               | Improve stability and interpretability by removing highly correlated features.                  |
| Top-k Important Features             | Build lightweight models with only the top contributing features.                               |

---

## 🤖 Model Selection Summary

| **Model**              | **Principle**                                                      | **Strengths**                                           | **Limitations**                                         |
|-----------------------|--------------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| Random Forest Regressor| Ensemble of decision trees with bagging to reduce variance.        | Handles non-linear, high-dimensional data; robust; interpretable. | May miss fine-grained patterns; memory intensive.     |
| Histogram GB Regressor | Boosting with histograms for efficiency.                           | High accuracy, fast, handles missing values.            | Sensitive to overfitting; needs careful tuning.        |
| MLP Regressor (Neural Net) | Fully connected network for regression tasks.                     | Captures complex, non-linear relationships; multi-output. | Requires careful scaling; prone to overfitting.        |

---

## 🏆 **Training Results Summary**

### Random Forest Regressor

| **Feature Set**        | **R²**  | **MAE**   | **RMSE**  |
|-----------------------|--------|----------|----------|
| 1 (19 features)       | 0.9754 | 0.0526   | 0.1187   |
| 2 (5 features)        | 0.9255 | 0.1037   | 0.2018   |
| 🥇 3 (73 features)     | **0.9936** | **0.0233** | **0.0584** |
| 4 (29 features)       | 0.9748 | 0.0536   | 0.1200   |
| 🥈 5 (88 features)     | 0.9932 | 0.0241   | 0.0603   |
| 6 (33 features)       | 0.9786 | 0.0454   | 0.1017   |
| 🥉 7 (10 features)     | 0.9902 | 0.0312   | 0.0758   |

---

### Histogram-Based Gradient Boosting Regressor

| **Feature Set**        | **R²**  | **MAE**   | **RMSE**  |
|-----------------------|--------|----------|----------|
| 1 (19 features)       | 0.9675 | 0.0708   | 0.1364   |
| 2 (5 features)        | 0.9217 | 0.1094   | 0.2066   |
| 🥈 3 (73 features)     | 0.9873 | 0.0482   | 0.0840   |
| 4 (29 features)       | 0.9677 | 0.0710   | 0.1363   |
| 🥇 5 (88 features)     | **0.9874** | **0.0481** | **0.0835** |
| 6 (33 features)       | 0.9729 | 0.0690   | 0.1245   |
| 🥉 7 (10 features)     | 0.9802 | 0.0581   | 0.1054   |

---

### MLP Regressor

| **Feature Set**        | **R²**  | **MAE**   | **RMSE**  |
|-----------------------|--------|----------|----------|
| 1 (19 features)       | 0.9627 | 0.0760   | 0.1431   |
| 2 (5 features)        | 0.9262 | 0.1079   | 0.2010   |
| 🥇 3 (73 features)     | **0.9835** | **0.0500** | **0.0822** |
| 4 (29 features)       | 0.9624 | 0.0755   | 0.1430   |
| 🥈 5 (88 features)     | 0.9823 | 0.0502   | 0.0867   |
| 🥉 6 (33 features)     | 0.9695 | 0.0683   | 0.1219   |
| 7 (10 features)       | 0.9654 | 0.0733   | 0.1361   |

---

## ⚙️ **Grid Search Results on Feature Set 7**

| **Model**     | **R²**   | **MAE**    | **RMSE**   | **Best Parameters** |
|--------------|---------|-----------|-----------|---------------------|
| Hist GB      | 0.9921 (better) | 0.0354 (better) | 0.0650 (better) | `learning_rate=0.1, max_iter=700, max_leaf_nodes=127, min_samples_leaf=10, l2_regularization=0.01` |
| RF           | 0.9902 (same)   | 0.0335 (worst)  | 0.0756 (better) | `n_estimators=200, max_features='sqrt', min_samples_leaf=1, min_samples_split=2` |
| MLP          | 0.9785 (better)  | 0.0546 (better) | 0.0962 (better) | `hidden_layer_sizes=(200, 100, 50), learning_rate_init=0.001, max_iter=1000, early_stopping=True` |

---

6. **06_predict_test_data.ipynb**  
   Preprocess test data, generate predictions using feature subset 3 and trained random forest reg, and save to `results/predicted_forces.xlsx`.

7. Link to my presentation slides -> (https://docs.google.com/presentation/d/1ozolS0Tft3jiP4LAy3abX7pH2Nz3lDLtm4BMeP4ylEw/edit?usp=sharing)


