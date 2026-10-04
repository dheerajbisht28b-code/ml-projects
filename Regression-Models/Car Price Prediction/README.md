# 🚗 Car Price Prediction using Tuned Decision Tree
## 📖 Project Overview
Predicting the selling price of a used car is a complex problem due to non-linear depreciation rates and extreme outliers (luxury cars). This project builds an end-to-end Machine Learning pipeline and an interactive web application to predict car prices accurately while ensuring the model is robust and generalized.

---

##  The Machine Learning Journey
This project isn't just about training a model; it's about solving the business problem of overfitting and interpretability.

### Phase 1: Linear Regression (Baseline)
- *Performance:* Achieved an R² score of *83.13%*.
- *Limitation:* The model struggled with high-value outliers, resulting in a high RMSE. While a Log Transformation improved the statistical score, it made the output (log prices) unintuitive for real-world business users.

### Phase 2: Default Decision Tree Regressor
- *Why switch?* Decision Trees naturally handle non-linear relationships and are invariant to feature scaling.
- *The Problem:* The default tree memorized the training data, leading to severe *overfitting* (Training R²: *99.7%* vs. Testing R²: *91.3%*).

### Phase 3: Hyperparameter Tuning (The Fix)
- Instead of blindly using GridSearch, I plotted a *Validation Curve* (Tree Depth vs. R² Score) to visually identify the "sweet spot."
- By tuning max_depth to *7*, the model learned the underlying patterns without memorizing noise.
- *Result:* The overfitting gap was successfully reduced from *8.4% down to just 0.2%*!
  - *Training R²:* 91.41%
  - *Testing R²:* 91.21%

---

## 🌐 Streamlit App Features
The model is deployed via a multi-tab, interactive Streamlit application designed for both end-users and technical stakeholders:

1. *🔮 Predict Tab:* Users input car specifications to get an instant price prediction. Includes a custom *Gauge Chart* to show how the predicted price compares to the overall market average.
2. * Market Analytics:* Interactive Plotly charts to explore market trends, brand pricing, and mileage vs. price clusters.
3. * Model Performance:* A dedicated tab showcasing the *Validation Curve*, proving to stakeholders that the model is robust and not overfitting.
4. *️ About:* Details about the tech stack and the modeling journey.

---

## ️ Tech Stack
- *Programming:* Python
- *Machine Learning:* Scikit-Learn (DecisionTreeRegressor, ColumnTransformer, Pipeline)
- *Data Manipulation:* Pandas, NumPy
- *Visualization:* Plotly, Matplotlib
- *Deployment:* Streamlit, Joblib

---

## 🚀 How to Run Locally

1. Clone the repository:
bash
   git clone https://github.com/dheerajbisht28b-code/ml-projects.git
   cd ml-projects/Regression-Models/Car Price Prediction


2. Install the required dependencies:
bash
   pip install -r requirements.txt


3. Run the Streamlit application:
bash
   streamlit run app.py


---

## 🔗 Links & Resources
- *Live Application:* [https://ml-projects-u5s7mywtbzs2rrdf9hiows.streamlit.app/]
- *Dataset:* CarDekho Dataset (Kaggle)

---
Built by Dheeraj | Portfolio Project for ML Internship Preparation
