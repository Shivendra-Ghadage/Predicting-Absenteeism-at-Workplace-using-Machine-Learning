# Predicting Absenteeism at Workplace Using Machine Learning

## 📌 Project Overview

Employee absenteeism can affect productivity, workforce planning, and operational efficiency. This project uses Machine Learning to analyze employee absenteeism patterns and predict whether an employee is likely to experience **excessive absenteeism**.

The project follows an end-to-end Machine Learning workflow:

**Data Preprocessing → Feature Engineering → Logistic Regression → Model Evaluation → Model Serialization → New Data Prediction → Visualization**

The project uses Python and Scikit-learn for data processing and Machine Learning, along with Tableau for visualizing prediction results.

---

## 🎯 Project Objectives

- Analyze employee absenteeism data.
- Understand the factors associated with absenteeism.
- Clean and preprocess the raw dataset.
- Perform feature engineering.
- Convert absenteeism hours into a binary classification target.
- Build a Logistic Regression classification model.
- Evaluate model performance.
- Save the trained model and scaler.
- Apply the trained model to new/unseen data.
- Generate absenteeism probabilities and predictions.
- Visualize prediction results using Tableau.

---

## 🏢 Business Problem

Employee absenteeism can create challenges for organizations in areas such as:

- Workforce planning
- Employee scheduling
- Productivity management
- Resource allocation
- Operational planning

The goal of this project is to build a predictive model that can identify observations associated with **excessive absenteeism** based on employee and workplace-related attributes.

> **Note:** In this project, "excessive absenteeism" is defined using the median value of `Absenteeism Time in Hours` in the training dataset.

---

# 📊 Dataset

The project uses an employee absenteeism dataset containing information related to:

- Absence reasons
- Date of absence
- Transportation expense
- Distance to work
- Age
- Daily workload
- Body Mass Index (BMI)
- Education
- Number of children
- Number of pets
- Absenteeism time in hours

### Raw Dataset

The original dataset contains:

- **700 records**
- **12 columns**

### Main Columns

| Column | Description |
|---|---|
| `ID` | Employee/observation identifier |
| `Reason for Absence` | Reason associated with the absence |
| `Date` | Date of absence |
| `Transportation Expense` | Employee transportation expense |
| `Distance to Work` | Distance between residence and workplace |
| `Age` | Employee age |
| `Daily Work Load Average` | Average daily workload |
| `Body Mass Index` | Employee BMI |
| `Education` | Education category |
| `Children` | Number of children |
| `Pets` | Number of pets |
| `Absenteeism Time in Hours` | Duration of absenteeism |

---

# 🔄 Project Workflow

```text
                Raw Dataset
                     │
                     ▼
            Data Preprocessing
                     │
                     ▼
            Feature Engineering
                     │
                     ▼
             Target Creation
                     │
                     ▼
              Data Scaling
                     │
                     ▼
            Train/Test Split
                     │
                     ▼
          Logistic Regression
                     │
                     ▼
             Model Evaluation
                     │
                     ▼
          Save Model + Scaler
                     │
                     ▼
             New Employee Data
                     │
                     ▼
                Prediction
                     │
                     ▼
            Prediction Results
                     │
                     ▼
               Visualization
```
---

## 🧹 Data Preprocessing

The raw dataset is processed before applying Machine Learning.

### Main Preprocessing Steps

1. Load the raw dataset.
2. Inspect the dataset structure and data types.
3. Check for missing values and duplicate records.
4. Remove the `ID` column from model inputs.
5. Process the `Reason for Absence` variable.
6. Group absence reasons into broader categories.
7. Convert the `Date` column into useful features.
8. Extract the month from the date.
9. Extract the day of the week.
10. Simplify the `Education` variable.
11. Prepare the final features for Machine Learning.
12. Save the preprocessed dataset.

---

## 🔎 Feature Engineering

Feature engineering is performed to transform the raw variables into meaningful features for the Machine Learning model.

### 1. Reason for Absence

The original `Reason for Absence` variable contains multiple categorical reason codes.

These reasons are grouped into four broader categories:

| Feature | Description |
|---|---|
| `Reason 1` | Disease-related reasons |
| `Reason 2` | Pregnancy-related reasons |
| `Reason 3` | Poisoning and related reasons |
| `Reason 4` | Other/light absence reasons |

The categorical reason values are converted into binary features and then grouped into broader categories.

### 2. Date Features

The original `Date` column is transformed into:

- `Month Value`
- `Day of the Week`

The original `Date` column is then removed from the model input.

### 3. Education

The education variable is simplified into two categories:

```text
0 → High School
1 → Higher Education
```
---

## 🎯 Target Variable

The original dataset contains the `Absenteeism Time in Hours` column.

Instead of predicting the exact number of absenteeism hours, the project converts the problem into a **binary classification problem**.

The median value of `Absenteeism Time in Hours` is used as the classification threshold.

```text
Absenteeism Time > Median
        ↓
Excessive Absenteeism = 1
```
---

## 🤖 Machine Learning Model

The project uses **Logistic Regression** as the machine learning algorithm.

Logistic Regression is suitable because the target variable contains two classes:

- `0` → Moderate Absenteeism
- `1` → Excessive Absenteeism

The model predicts both the class and the probability of excessive absenteeism.

### Model Development Process

1. Split the dataset into training and testing sets.
2. Apply feature scaling to numerical features.
3. Train the Logistic Regression model.
4. Evaluate the model on training and testing data.
5. Analyze model coefficients and odds ratios.
6. Remove selected features based on the model analysis.
7. Train the final Logistic Regression model.

---

## ⚙️ Feature Scaling

Since the numerical features have different ranges, **StandardScaler** is used to standardize the numerical variables.

The following features are scaled:

- Month Value
- Day of the Week
- Transportation Expense
- Distance to Work
- Age
- Daily Work Load Average
- Body Mass Index
- Children
- Pets

Binary and dummy variables are kept unscaled.

The same scaler is saved and reused when making predictions on new data to ensure consistency between training and prediction.

---

## 📈 Model Development

The dataset is divided into training and testing datasets using an **80:20 split**.

```python
train_test_split(
    scaled_inputs,
    targets,
    test_size=0.2,
    stratify=targets,
    random_state=1
)
```

---

## 🙏 Acknowledgements

This project was developed as a practical machine learning project to understand the complete workflow of:

- Data preprocessing
- Feature engineering
- Machine learning model development
- Model evaluation
- Model serialization
- Prediction on new data
- Data visualization

The project helped in applying Python and Machine Learning concepts to a workplace-related dataset.

---

## ⚠️ Disclaimer

This project is created for **educational and portfolio purposes**.

The predictions generated by the model are based on patterns present in the provided dataset. They should not be used as the sole basis for making actual employee-related decisions.

The term **"Excessive Absenteeism"** in this project refers specifically to absenteeism hours greater than the median value of the training dataset. It does not represent a universal workplace or HR threshold.

---

## 📌 Project Summary

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Target Creation
     ↓
Feature Scaling
     ↓
Train / Test Split
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
New Employee Data
     ↓
Prediction
     ↓
Visualization
```

---

## 👨‍💻 Author

Shivendra Ghadage

Bachelor of Engineering – Artificial Intelligence & Data Science

Interested in:

Data Analytics
Data Science
Machine Learning
SQL
Power BI
Python

---

### ⭐ Thank You for Visiting!

Thank you for taking the time to explore this project.

If you found this project useful or interesting, consider giving the repository a ⭐.

**Built with Python & Machine Learning.**
