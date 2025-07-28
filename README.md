# MLAdditive: Predicting Pipe Lifetime

Welcome to **MLAdditive**, a machine learning application designed to predict the lifetime of industrial pipes using material composition and testing parameters. This repository features:

- ✅ A public **Streamlit web app** for lifetime prediction using a trained model.
- 🔐 A private **training pipeline** for model development on sensitive datasets.
- 📊 Interpretable performance results of the trained model.
- 📂 Modular and clean project architecture.

---

## Project Overview

Industrial pipe lifetime prediction is vital for maintenance and failure prevention. Our solution uses supervised regression models trained on proprietary ISO9080-based datasets to estimate lifetime in a few seconds.

---

## Model Performance & Results

We evaluated multiple models. The **Random Forest Regressor** achieved the best generalization performance. Below are the test results and visualizations:

### Evaluation Metrics (Test Set)

| Model               | MAE     | MSE     | RMSE    | R²     |
|--------------------|---------|---------|---------|--------|
| Ridge Regression   | 21.23   | 577.30  | 24.03   | 0.578  |
| Random Forest      | **2.24**| **47.09**| **6.86**| **0.985** |

### Visual: True vs Predicted (Random Forest)

![Random Forest Prediction](results/regression/random_forest/true_vs_pred_rf.png)

The plot demonstrates high predictive accuracy with minimal error across the test set.

---

## 📦 Deployment & Inference Instructions

Anyone can run the Streamlit app for inference using the pre-trained model (`best_rf_model.pkl`):

```bash
# Clone the repository
git clone https://github.com/HJ701/MLAdditive.git
cd MLAdditive

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app/app.py
