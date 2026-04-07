#  Industrial-Anomaly-Predictor

An end-to-end Machine Learning pipeline designed to predict industrial equipment failure before it happens.

###  Tech Stack
* **Python / Pandas:** Data manipulation.
* **Scikit-Learn:** Core ML algorithms.
* **SHAP:** Model explainability and interpretability.
* **Joblib:** Model serialization for production.

###  Methodology
1. **Feature Engineering:** Creation of a custom `Stress Index` to capture sensor correlations.
2. **Pre-processing:** Automated scaling and outlier handling.
3. **Training:** Random Forest Classifier with optimized hyperparameters.
4. **Interpretability:** Local explanation of specific failure predictions using SHAP values.

###  How to Run
```bash
python main.py
