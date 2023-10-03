# AI-Driven Credit Risk Prediction Engine

## Description
An AI-driven credit risk prediction model developed using deep learning techniques to evaluate the creditworthiness of individual and institutional clients. The model enhances loan processing capabilities, reduces processing time, and improves prediction accuracy, leading to more informed credit decisions.

## Data Description

The dataset used for this project is comprehensive and includes a variety of features essential for evaluating the creditworthiness of clients. Below is a breakdown of the features and their descriptions:

### Features:

- **person_age:** Age of the individual.
- **person_income:** Annual income.
- **person_home_ownership:** Type of home ownership (e.g., rent, own).
- **person_emp_length:** Employment length in years.
- **loan_intent:** The intent of the loan (e.g., education, medical expenses).
- **loan_grade:** The grade of the loan.
- **loan_amnt:** The loan amount.
- **loan_int_rate:** The interest rate on the loan.
- **loan_status:** Loan status (0 for non-default, 1 for default).
- **loan_percent_income:** The loan amount as a percentage of annual income.
- **cb_person_default_on_file:** Whether the person has a history of default (‘Y’ for yes, ‘N’ for no).
- **cb_person_cred_hist_length:** The length of the person’s credit history.

### Data Preprocessing:

The following preprocessing steps were applied to clean and transform the raw data into a usable format for model training:

1. **Handling Missing Values:**
   - Filled numerical missing values with the mean or median.
   - Categorical missing values were filled with the mode or removed.

2. **Encoding Categorical Variables:**
   - Converted categorical variables like ‘cb_person_default_on_file’ and ‘person_home_ownership’ into numerical form using encoding techniques.

3. **Feature Scaling:**
   - Applied Min-Max Scaling or Standard Scaling to ensure all features have a similar scale, improving model performance.

4. **Feature Engineering:**
   - Created new features or optimized existing ones to enhance the model’s predictive performance.

5. **Handling Imbalanced Data:**
   - Implemented techniques like oversampling, undersampling, or generating synthetic samples to balance the dataset, ensuring that the model is not biased towards the majority class.

These preprocessing steps ensured that the data is clean, balanced, and ready for training the deep learning model, leading to more accurate and reliable predictions.

## Features
- **Hybrid Model:** Combines convolutional neural networks and recurrent neural networks for enhanced prediction accuracy.
- **Interpretability:** Integrates SHAP (SHapley Additive exPlanations) for clear insights into model decisions, ensuring regulatory compliance and trust.

## Installation
Clone the repository and install the required packages using the following commands:

```bash
git clone https://github.com/yourusername/yourrepositoryname.git
cd yourrepositoryname
pip install -r requirements.txt
