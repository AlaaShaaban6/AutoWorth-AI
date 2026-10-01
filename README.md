# AutoWorth AI — Used Car Price & Deal Advisor

## Project Overview
AutoWorth AI is a machine learning regression project for predicting used-car prices and providing a data-driven deal assessment.

The project uses the **100,000 UK Used Car Dataset** covering Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, and Volkswagen.

The target variable is `Price`.

The project extends price prediction with a Smart Deal Advisor:
- More than 10% below predicted price → Great Deal
- 5% to 10% below → Good Deal
- Within ±5% → Fair Price
- More than 5% to 10% above → Slightly Overpriced
- More than 10% above → Overpriced

## Objectives
1. Integrate manufacturer datasets.
2. Investigate and clean data quality issues.
3. Perform exploratory data analysis.
4. Engineer meaningful features.
5. Compare regression models.
6. Analyze overfitting and generalization.
7. Tune the selected model.
8. Evaluate it on an untouched test set.
9. Analyze feature importance and large errors.
10. Build a Smart Deal Advisor.
11. Deploy the model with Streamlit.

## Dataset
**Source:** Kaggle — 100,000 UK Used Car Dataset  
https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

Two explicitly unclean files (`unclean focus.csv` and `unclean cclass.csv`) were excluded.

## Data Preparation
Initial combined dataset: **108,540 rows × 10 columns**.

Key findings:
- 9,353 missing `tax` values
- 9,353 missing `mpg` values
- 2,273 exact duplicate rows
- 3 suspicious year records (1970 and 2060)
- 286 initial `engineSize = 0` values

Cleaning actions:
- Removed 2,273 exact duplicates.
- Removed 3 invalid year records.
- Converted 280 zero engine-size values to missing after duplicate/year cleaning.
- Retained missing `tax` and `mpg` for later imputation.
- Retained plausible extreme prices and mileage.
- Retained unusual MPG values when not proven invalid.

Final cleaned dataset: **106,264 rows × 10 columns**.

## Exploratory Data Analysis
The project examined price distribution, price vs. mileage, price vs. year, average price by manufacturer, price by transmission, price by fuel type, and numerical correlations.

Important observations:
- Prices are strongly right-skewed, with most cars below £20,000.
- Higher mileage is generally associated with lower prices.
- Newer vehicles generally have higher prices.
- Mercedes, Audi, and BMW have higher average prices in this dataset.
- Transmission and fuel type show noticeable differences in price distributions.

Approximate correlations with price:
| Feature | Correlation |
|---|---:|
| Engine Size | 0.64 |
| Year | 0.50 |
| Mileage | -0.43 |
| Tax | 0.31 |
| MPG | -0.30 |

Correlation represents linear association, not causation.

## Train / Validation / Test
The data was split before fitting preprocessing:
- 70% training
- 15% validation
- 15% test

The test set remained untouched until final evaluation.

### Feature Engineering
Three features were created:
- `car_age = 2021 - year`
- `mileage_per_year = mileage / car_age`
- `engine_mpg_ratio = engineSize / mpg`

### Preprocessing
Numerical features used median imputation and StandardScaler.

Categorical features used most-frequent imputation and One-Hot Encoding with `handle_unknown="ignore"`.

The preprocessing pipeline was fitted only on the training set.

## Models
Five regression models were evaluated:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor

### Model Comparison
| Model | Train R² | Validation R² | Validation MAE | Validation RMSE |
|---|---:|---:|---:|---:|
| Random Forest | 0.9937 | **0.9569** | **£1,175.65** | **£2,048.15** |
| XGBoost | 0.9468 | 0.9374 | £1,569.29 | £2,468.80 |
| Decision Tree | 0.9995 | 0.9349 | £1,453.62 | £2,516.55 |
| Gradient Boosting | 0.9023 | 0.8966 | £2,108.96 | £3,172.03 |
| Linear Regression | 0.8685 | 0.8754 | £2,188.83 | £3,482.08 |

Random Forest had the strongest validation performance. Decision Tree showed a larger train-validation gap.

## Hyperparameter Tuning
RandomizedSearchCV with 3-fold cross-validation evaluated six sampled Random Forest configurations.

Best configuration:
```text
n_estimators = 100
max_depth = 20
max_features = 0.5
min_samples_split = 2
min_samples_leaf = 1
```

Validation performance:
- R² improved from **0.9569** to **0.9578**
- MAE improved from **£1,175.65** to **£1,173.53**
- RMSE improved from **£2,048.15** to **£2,026.45**

Training R² decreased from 0.9937 to 0.9862, reducing the train-validation gap.

## Final Model
The tuned Random Forest was evaluated once on the untouched test set.

| Metric | Test Result |
|---|---:|
| R² | **0.9652** |
| MAE | **£1,164.69** |
| RMSE | **£1,823.59** |

The model explains approximately 96.5% of the variation in test-set prices. MAE is approximately £1,165.

## Model Understanding
Top grouped Random Forest feature importances:
| Feature | Importance |
|---|---:|
| Engine-MPG Ratio | 0.2323 |
| Engine Size | 0.1671 |
| Transmission | 0.1590 |
| Car Age | 0.1119 |
| Year | 0.1052 |
| Mileage | 0.0623 |
| Model | 0.0621 |
| Make | 0.0348 |
| MPG | 0.0321 |
| Mileage Per Year | 0.0142 |
| Tax | 0.0123 |
| Fuel Type | 0.0066 |

These are relative predictive importance values within the Random Forest and should not be interpreted as causal effects.

## Large Error Investigation
Large errors included premium or relatively uncommon vehicles such as Audi R8, BMW X5, Mercedes C Class, Mercedes S Class, Volkswagen Caravelle, and Ford Mustang.

Examples:
- Audi R8: actual £125,000 vs. predicted approximately £95,112
- BMW X5: actual £39,948 vs. predicted approximately £13,352
- Ford Mustang: actual £32,990 vs. predicted approximately £52,638

These cases illustrate that errors can be larger for uncommon or high-priced vehicle segments.

## Smart Deal Advisor
The seller's asking price is compared with the predicted market price.

Example:
```text
Predicted Market Price: £15,682.47
Seller Price: £15,000.00
Difference: -£682.47
Difference %: -4.35%
Deal Rating: Fair Price
```

The advisor is a data-driven pricing aid, not a guarantee of actual market value.

## Streamlit Application
The application accepts:
- Make
- Model
- Year
- Mileage
- Transmission
- Fuel Type
- Engine Size
- Seller Price

It returns:
- Estimated Market Price
- Seller Price
- Difference
- Difference %
- Deal Rating

Saved deployment files:
- `tuned_random_forest.pkl`
- `preprocessor.pkl`
- `deployment_config.pkl`

## Project Structure
```text
AutoWorth-AI/
├── notebook/
│   └── AutoWorth_AI.ipynb
├── dataset/
├── autoworth_app/
│   ├── app.py
│   ├── tuned_random_forest.pkl
│   ├── preprocessor.pkl
│   └── deployment_config.pkl
├── README.md
└── presentation/
    └── AutoWorth_AI_Presentation.pptx
```

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib, Streamlit, Google Colab.

## Key Learning Outcomes
This project covers an end-to-end ML workflow: data integration, quality investigation, cleaning, EDA, feature engineering, leakage-safe preprocessing, model comparison, overfitting analysis, hyperparameter tuning, final evaluation, feature importance, error analysis, model serialization, and Streamlit deployment.

## Limitations
The dataset does not include factors such as vehicle condition, service history, number of owners, optional equipment, location, accident history, warranty, or market timing. These factors can affect real-world prices.

The Streamlit app does not request `tax` and `mpg`; the fitted preprocessing pipeline handles these missing inputs through imputation.

## Conclusion
AutoWorth AI demonstrates how a regression model can be transformed into a practical used-car pricing and deal-analysis application. The tuned Random Forest achieved **R² = 0.9652**, **MAE = £1,164.69**, and **RMSE = £1,823.59** on the untouched test set.
