# README: Hyperspectral Corn Data Analysis & Mycotoxin Prediction

This **Google Colab notebook** (`ML_task(1).ipynb`) implements a full end-to-end pipeline for analyzing hyperspectral corn sample data and predicting mycotoxin (vomitoxin) levels. Below is a step-by-step guide on what the notebook does and how to use it, along with brief justifications for each stage.

---

## 1. Notebook Overview

The notebook goes through these major stages:

1. **Data Import & Exploration**
2. **Outlier Analysis**
3. **Spectral Analysis**
4. **Feature Scaling & PCA**
5. **Regression Models (Linear, XGBoost, Random Forest)**
6. **Deep Learning Models (CNN, LSTM, Transformer)**
7. **Hyperparameter Tuning (RandomForest)**

We conclude with a summary of limitations and potential improvements.

---

## 2. Environment Setup

1. **Open** the notebook in Google Colab.
2. **Upload** your dataset (`TASK-ML-INTERN.csv`), or confirm that the path `/content/TASK-ML-INTERN.csv` is correct.
3. **Install** any missing dependencies in Colab (e.g., `!pip install xgboost` or `!pip install tensorflow`). The notebook might contain code cells for these installations.

---

## 3. Workflow Detailed

### A. Data Import & Exploration
- **Dataframe Creation**: Loads the CSV into a pandas DataFrame (`df`).
- **Shape/Head**: Verifies the dataset dimensions and inspects the first few rows.
- **Null Check**: `df.isnull().sum().sum()` tallies missing values.
- **Descriptive Stats**: Examines mean, std, min, max for each column.

### B. Outlier Analysis
- **Boxplots (Raw & Log)**: Visualizes potential outliers in vomitoxin data. A log transform (`log_vomitoxin_ppb`) is added to reduce skewness.

### C. Spectral Analysis
- **Identify Spectral Columns**: Excludes target columns (`vomitoxin_ppb`, `log_vomitoxin_ppb`) and any ID columns.
- **Average Spectrum**: Plots mean reflectance across wavelength bands.
- **Correlation Heatmap**: Investigates correlation between bands.
- **Reflectance Heatmap**: Visualizes sample-by-band data.

### D. Feature Scaling & PCA
- **Scaling**: Standardizes the feature matrix to remove unit disparities.
- **PCA (Explained Variance)**: Finds how many principal components account for ~97% of variance.
- **Dimensionality Reduction**: Uses those principal components to transform the data.
- **Linear Regression (Raw vs. Log)**: Demonstrates a simple baseline regression.

### E. Regression Models
1. **XGBoost**: Trains on both raw and log-transformed targets.
2. **RandomForest**: Similarly tested for raw and log targets.
3. **Metrics**: Reports RMSE and R² to compare.

### F. Deep Learning Models
All models are tested on **non-reduced** (original) and **PCA-reduced** data.
1. **Deep CNN** (Convolutional Neural Network)
   - Uses 1D convolutions to learn patterns in the spectral dimension.
   - Compares raw target vs. log target predictions.
2. **Deep LSTM** (Long Short-Term Memory)
   - Treats spectral data as a sequence.
   - Similarly reports MAE, RMSE, and R².
3. **Transformer**
   - Applies attention across the spectral sequence.
   - Evaluates raw and log target.

### G. Hyperparameter Tuning (RandomForest Example)
- **GridSearchCV**: Searches over `n_estimators`, `max_depth`, `min_samples_split`.
- **Best Params**: Prints the best parameter combination found.
- **Visualizations**: Plots log actual vs. log predicted scatter.

---

## 4. Usage Instructions
1. **Run Cells in Order**: Each section depends on previous steps for data transformations.
2. **Adjust Hyperparameters**: For each model, tweak epochs (deep models) or search ranges (RF, XGBoost).
3. **Interpret Results**: Look at MAE/RMSE/R² for each approach. Decide if raw or log target is superior.

---

## 5. Tips & Common Issues
- **Memory**: Large PCA transforms or deep learning can exceed Colab’s RAM if sample size or model size is large.
- **Convergence**: If models overfit or metrics are unstable, reduce complexity (layers, components) or apply regularization.
- **Real-World Data**: Sensor noise and data bias can significantly affect performance.

---

## 6. Conclusion

By following this notebook, you’ll:
1. Preprocess and scale hyperspectral data.
2. Reduce dimensionality via PCA.
3. Compare multiple regression approaches (linear, XGBoost, RandomForest, CNN, LSTM, Transformer).
4. Conduct hyperparameter tuning for at least one of these.

For further enhancements, consider:
- More extensive hyperparameter searches
- Additional deep architectures (e.g., deeper Transformers)
- Different scaling or feature-engineering techniques

**End of README**
