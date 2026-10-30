# Food Delivery Time Estimation

## Project Overview

Food Delivery Time Estimation is a Machine Learning project that predicts the estimated time required to deliver a food order. The system uses historical food delivery data and order-related features to predict delivery duration.

The project uses an **XGBoost Regressor** model and a **Streamlit** web application to provide interactive predictions, data visualizations, model performance analysis, and feature importance.

## Objectives

- Predict estimated food delivery time using machine learning.
- Analyze food delivery data and identify useful patterns.
- Understand the influence of traffic, distance, weather, and other available features.
- Evaluate model performance using regression metrics.
- Display feature importance and interactive visualizations.
- Provide a user-friendly dashboard for delivery time estimation.

## Technologies Used

| Technology | Purpose |

| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing and model evaluation |
| XGBoost | Regression model for delivery time prediction |
| Plotly | Interactive charts and visualizations |
| Streamlit | Interactive web application |
| Joblib | Loading the saved machine learning model |

## Project Features

### 1. Home
Provides an introduction to the project, its objectives, and how the system works.

### 2. Predict Delivery Time
Allows users to enter available order details and obtain an estimated delivery duration from the trained model.

### 3. Data Analysis
Displays charts and summaries to explore the food delivery dataset and understand patterns in the data.

### 4. Model Performance
Evaluates the trained regression model using metrics such as:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains variation in delivery time

### 5. Feature Importance
Displays the relative importance of input features according to the trained model.

### 6. About Project
Explains the project overview, objectives, technologies, workflow, and expected outcome.

## Machine Learning Model

This project uses the **XGBoost Regressor**, a gradient-boosting model suitable for regression tasks.

The model learns patterns from historical delivery records and uses the available input features to estimate delivery time for new orders.

The dataset's `Delivery Time` column is the target variable. Identifier columns such as `ID` and `Delivery Person ID` are excluded from model inputs when preparing the training data.

## Project Workflow

1. Collect and load the food delivery dataset.
2. Inspect and preprocess the data.
3. Separate input features and the target variable.
4. Train and evaluate the XGBoost regression model.
5. Save the trained model.
6. Load the saved model into Streamlit.
7. Allow users to enter order details.
8. Display predicted delivery time and analysis visualizations.

## Project Structure

Food Delivery Time Estimation/
│
├── app.py
├── style.css
├── README.md
│
├── dataset/
│   └── foodtimedataset.csv
│
└── pickle_file/
    └── food_delivery_xgboost.pkl


*Note: The structure above assumes your model and dataset use these filenames and locations.*

## Installation

### Step 1: Install Python

Install a compatible version of Python on your computer.

### Step 2: Open the project folder

Open the project folder in VS Code or a terminal.

### Step 3: Install the required libraries

Run the following command:

pip install streamlit pandas numpy scikit-learn xgboost plotly joblib streamlit-option-menu
```

### Step 4: Check the project files

Ensure that the dataset and trained model are present at the paths expected by `app.py`.

### Step 5: Run the application

python -m streamlit run app.py

The application will open in your web browser.

## Dataset

The project uses `foodtimedataset.csv`, containing historical food delivery information.

Depending on the actual dataset, input features may include:

- Restaurant preparation time
- Delivery distance
- Traffic conditions
- Weather conditions
- Order-related information
- Other available delivery-related features

The target variable is "Delivery Time".

## Expected Outcome

The application provides an interactive platform for estimating food delivery duration, exploring delivery data, evaluating model performance, and understanding model feature importance.

Predictions are estimates based on the patterns learned from the training data and may differ from actual delivery times.

## Future Enhancements

- Integrate real-time traffic and weather data.
- Update predictions as an order progresses.
- Add customer order tracking.
- Retrain the model with more recent delivery records.
- Deploy the application online for wider access.

## Conclusion

The Food Delivery Time Estimation project demonstrates how machine learning can be applied to a practical delivery-time prediction problem. By combining an XGBoost regression model with a Streamlit dashboard, it provides predictions and visual analysis through a simple, interactive interface.



**Project Title:** Food Delivery Time Estimation  
**Domain:** Machine Learning and Data Science  
**Model:** XGBoost Regressor  
**Interface:** Streamlit