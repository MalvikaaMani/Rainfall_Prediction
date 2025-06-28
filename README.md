# Rainfall_Prediction
This project develops a machine learning model to predict daily rainfall based on historical weather data. It leverages a Random Forest Classifier with robust data preprocessing, feature engineering, and hyperparameter tuning to achieve accurate predictions, addressing common challenges like class imbalance.
<h1>
  Project Overview
</h1>
The primary goal of this project is to build an effective rain prediction system. By analyzing various meteorological parameters recorded throughout the day, the model learns patterns associated with rainfall events. The output is a binary classification: whether it will rain on a given day (1) or not (0).
<h2>Dataset Description</h2>
The model is trained on historical weather observations provided in testset.csv. The columns of the dataset has been added in this repository under dataset_columns.png file.

<h3> Features</h3>
The raw sub-daily data is transformed into a daily aggregated dataset, which forms the basis for model training.
<br>
Derived Features
<br>
For each day, the following aggregate features are created from the raw data:
<p> Numerical Features (e.g., temperature, humidity, pressure, wind speed, visibility, dew point, precipitation, heat index, wind chill):</p>
<ul>
  <li>Mean (_mean)</li>
  <li>Maximum (_max)</li>
  <li>Minimum (_min)</li>
  <li>Standard Deviation (_std)</li>
</ul>
<p> Binary Flag Features (e.g., fog, hail, snow, thunder, tornado):</p>
<ul>
  <li>Maximum (_occurred): Indicates if the event occurred at least once during the day.</li>
</ul>
<p>Target Variable:</p>
<ul>
  <li>rain_daily: A binary flag (1 or 0) indicating whether it rained at all on that specific day, derived from the maximum of _rain observations for the day.</li>
</ul>
Missing values are handled using median imputation for numerical features and mode imputation for categorical/binary features before aggregation. A final imputation step ensures no NaNs remain in the feature set X.
<h3>Technologies Used</h3>
<ul>
  <li>Python 3.x</li>
  <li>pandas: For data manipulation and analysis.</li>
  <li>scikit-learn: For machine learning algorithms, preprocessing, model selection, and evaluation.</li>
  <li>imbalanced-learn (imblearn): For handling imbalanced datasets using SMOTE.</li>
  <li>matplotlib: For creating static, interactive, and animated visualizations (e.g., confusion matrix, ROC curve).</li>
  <li>NumPy: For numerical operations.</li>
  <li>SciPy: For statistical functions used in hyperparameter distributions (randint).</li>
</ul>
<h3> Results</h3>
The model's performance metrics on the test set, after hyperparameter tuning and handling class imbalance, are presented below:
<ul>
  <li> Best cross validation accuracy for Random Forest: 0.9003</li>
  <li> Test Accuracy: 0.9101</li>
  <li> AUC Score: 0.9305</li>
</ul>

