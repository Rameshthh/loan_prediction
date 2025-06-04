# 🏦 Loan Approval Prediction using Machine Learning

This project is a supervised machine learning model that predicts whether a loan will be approved or not based on historical applicant data. It is built using Python and various data science libraries including Pandas, Matplotlib, and Scikit-learn.

## 📁 Dataset

The dataset includes anonymized data of loan applicants:
- `trainloan.csv` - Training dataset with loan approval status
- `testloan.csv` - Test dataset without labels

### Features include:
- Gender, Marital Status, Education, Self Employment
- Applicant and Co-applicant Income
- Loan Amount and Loan Term
- Credit History
- Property Area

## 🔍 Problem Statement

To build a machine learning model that can accurately classify whether a loan application will be **Approved (Y)** or **Not Approved (N)** based on the applicant’s profile.

## 🧠 ML Approach

1. **Data Preprocessing**:
   - Handled missing values
   - Label Encoding for categorical variables
   - Outlier detection and treatment

2. **Exploratory Data Analysis (EDA)**:
   - Visualized trends based on loan status, gender, income, and credit history

3. **Model Training**:
   - Used Logistic Regression as the baseline model
   - Accuracy Score and Confusion Matrix for evaluation

4. **Prediction**:
   - Final predictions were made on the test dataset

## ✅ Results

- Model Accuracy (on train data): *~78%*
- Used Logistic Regression due to its interpretability and decent performance on small, tabular datasets.

## 🛠 Tools & Technologies

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook


