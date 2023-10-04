# Deep Learning for Credit Risk Prediction

## Overview

This repository contains a machine learning project that aims to predict credit risk using various machine learning techniques, including deep learning and ensemble methods. The project covers the entire machine learning workflow from data preprocessing, feature engineering, model building, to evaluation.

## Dataset

The dataset used in this project is the `credit_risk_dataset.csv`, which contains information about credit applicants, including their personal details, loan details, and credit histories. The target variable is the `loan_status`, indicating whether an applicant is likely to default on their loan or not.

### Features

- `person_age`: The age of the credit applicant
- `person_income`: The income of the applicant
- `person_home_ownership`: The home ownership status of the applicant
- `person_emp_length`: Employment length in years
- `loan_intent`: The intent of the loan
- `loan_grade`: The grade of the loan
- `loan_amnt`: The loan amount
- `loan_int_rate`: The interest rate on the loan
- `loan_percent_income`: The loan amount as a percentage of income
- `cb_person_default_on_file`: Whether the person has defaulted in the past
- `cb_person_cred_hist_length`: The credit history length of the applicant

### Data Preprocessing

- Handling missing values by dropping rows with null values
- Encoding categorical variables into numerical representations
- Standardizing features to have a mean of 0 and a standard deviation of 1

## Dependencies

- pandas
- numpy
- scikit-learn
- tensorflow
- shap
- matplotlib (optional for visualizations)

## How to Run

1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Run the Jupyter Notebook to execute the code cells in order.

## Methodology

### Model Building

The project employs a variety of models, including:

1. **Support Vector Machine (SVM)**: 
   - Hyperparameters are tuned using RandomizedSearchCV.
   - The model is trained with the best parameters.

2. **Deep Learning**: 
   - A Sequential model with multiple dense layers, batch normalization, and dropout for regularization.
   - The model is compiled and trained using the Adam optimizer and binary cross-entropy loss.

3. **Random Forest**: 
   - Implemented as an ensemble method to improve prediction performance.

### Feature Engineering

- Polynomial features are created to explore complex relationships and interactions among features.

### Model Interpretability

- SHAP (SHapley Additive exPlanations) is used to understand the impact of different features on the model’s predictions.

### Evaluation

- The models are evaluated using accuracy, confusion matrix, and classification report to measure their performance.

## Results

The results indicate a satisfactory level of accuracy in predicting credit risk, showcasing the potential of machine learning in enhancing decision-making in credit allocations. Detailed results and visualizations can be found in the Jupyter Notebook.

## License

This project is open-source and available to anyone under the [MIT License](LICENSE).
