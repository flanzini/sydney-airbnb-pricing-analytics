# Airbnb Price Prediction – Sydney (QBUS6810 Kaggle Competition)

This repository contains the code for a group project completed for **QBUS6810** at the University of Sydney, where we participated in a Kaggle-style machine learning competition to **predict nightly Airbnb listing prices** from listing metadata.

Our group ranked **2nd overall** in the final leaderboard.

---

## Project Overview

The goal of the competition was to build a regression model that predicts the **nightly price** of Airbnb listings using:

- Host information  
- Property type and room type  
- Location features  
- Availability and review metrics  
- Text features from listing descriptions / titles  

This repository contains the main Jupyter notebook we used for:

1. Exploratory Data Analysis (EDA)  
2. Data cleaning and feature engineering  
3. Model training & hyperparameter tuning  
4. Stacking and final predictions for the competition submission

---

## Repository Structure

```text
.
├─ notebooks/
│  └─ airbnb_price_prediction_sydney.ipynb   # main notebook used in the competition
├─ data/
│  ├─ .gitkeep                               # placeholder; put raw/train/test data here
├─ src/
│  ├─ __init__.py
│  └─ features.py                            # (optional) feature engineering helpers
├─ README.md
├─ requirements.txt
└─ .gitignore
```

> **Note:** The original Kaggle dataset is **not** included for size and licensing reasons.  
> Place the downloaded `train.csv` / `test.csv` files in the `data/` folder and update the paths in the notebook if needed.

---

## Data & Features

### Raw data

The original data was provided through the course Kaggle competition and contains (among others):

- **Listing metadata**: room type, property type, accommodates, bathrooms, bedrooms, beds, etc.  
- **Host/booking information**: host response rate, host listings count, availability, cancellation policy.  
- **Location**: latitude, longitude.  
- **Reviews**: number of reviews, review scores.  
- **Text**: description / name of the listing.

The project focuses on **Sydney**, and some geospatial features are built around the **distance to the CBD**.

### Cleaning & preprocessing

In the notebook we:

- Handle missing values for:
  - **Numeric variables** (e.g., filling with median/mean or domain-based values).
  - **Categorical variables** (e.g., filling with explicit `"Missing"` category).
- Convert monetary and percentage fields to numeric, e.g.:
  - `price` (removing currency symbols and commas).
  - `response_rate` (parsing `"85%"` → `0.85`).
- Remove obvious outliers (e.g. extremely high prices or listings with unrealistic values).

### Feature engineering

Some of the main engineered features include:

- **Geospatial features**
  - Distance from each listing to Sydney CBD using latitude/longitude.
- **Aggregated / transformed numeric features**
  - Log/Power transforms and scaling for skewed variables.
- **Text-based features**
  - Simple NLP preprocessing with `nltk`:
    - Tokenization
    - Lowercasing, punctuation removal
    - Stopword removal
    - Stemming
  - Counts of “positive” description tokens (e.g. words like *apartment*, *bedroom*, *location*, *beach*, *spacious*, *modern*) to capture listing quality.

---

## Modeling Approach

We implemented and compared several regression models using **scikit-learn**, **XGBoost**, and **LightGBM**.

### Base models

- **Linear Regression**
- **Regularized Linear Models (LassoCV, RidgeCV)**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- **LightGBM Regressor**

All models were trained on the processed feature set, with hyperparameters tuned using validation splits / randomized search where appropriate.

### Stacking

We used the `vecstack` library to perform **stacking**:

1. Train multiple base learners on the training folds.  
2. Use out-of-fold predictions as features for a **meta-model**.  
3. Train the meta-model (e.g., a linear model or tree-based model) on these stacked features.

This approach helped us capture diverse patterns in the data and improved our leaderboard performance.

---

## Evaluation

We evaluated models using metrics such as:

- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² score**

The final stacked model was selected based on validation performance and then used to generate predictions for the competition test set, which achieved **2nd place** on the leaderboard.

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/airbnb-price-prediction-qbus6810.git
   cd airbnb-price-prediction-qbus6810
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # on Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the data**

   - Download the `train.csv` and `test.csv` from the course Kaggle competition or LMS.
   - Save them in the `data/` folder, e.g.:
     - `data/train.csv`
     - `data/test.csv`

   Update the file paths in `notebooks/airbnb_price_prediction_sydney.ipynb` if necessary.

5. **Run the notebook**

   ```bash
   jupyter notebook notebooks/airbnb_price_prediction_sydney.ipynb
   ```

---

## Dependencies

Main Python libraries used:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `wordcloud`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `vecstack`
- `nltk`
- `basemap`

See [`requirements.txt`](./requirements.txt) for a full list.

---

## Acknowledgements

- **Course**: QBUS6810 (University of Sydney)  
- **Teammates**: Project Group 36  
- **Competition host**: Course staff via Kaggle in-class competition
