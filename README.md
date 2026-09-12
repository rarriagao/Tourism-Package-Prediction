# Tourism-Package-Prediction

## 📋 Business Context


**"Visit with Us"** is a leading travel company aiming to transform its customer engagement strategy through data-driven decision-making. With the launch of a new **Wellness Tourism Package**, the organization faces challenges in accurately identifying customers who are most likely to purchase the package.

The existing manual customer selection process is inefficient, inconsistent, and prone to human error, resulting in missed sales opportunities and ineffective marketing campaigns. To overcome these challenges, the company seeks to implement a **scalable, automated, and reliable predictive system** that enables precise customer targeting and continuous model improvement.

By adopting an **MLOps-driven approach**, this project integrates data preprocessing, machine learning model development, deployment, and CI/CD automation using **GitHub Actions**, ensuring operational efficiency and adaptability to evolving customer behavior.

## 🚀 Live Application

- **Streamlit**: [View Live](https://tourism-package-prediction-couk27ovcnqzkqtqcg4d22.streamlit.app/)

## 🗂️ GitHub Structure

```
tourism_project/
├── .github/
│   └── workflows/
│       └── pipeline.yml                 # GitHub Actions CI/CD workflow
├── data/
│   └── tourism.csv                      # Original cleaned dataset
├── deployment/
│   ├── app.py                           # Streamlit web application
│   ├── best_tourism_model_v1.joblib     # Packed best model
│   └── requirements.txt                 # Deployment dependencies
├── model_building/
│   ├── data_register.py                 # Dataset registration
│   ├── prep.py                          # Data preprocessing script
│   └── train.py                         # Model training with MLflow tracking
└── requirements.txt                     # Workflow dependencies
```

## 🎯 Pipeline Stages

The MLOps pipeline automates the following stages:

1. Data ingestion and validation
2. Data cleaning and preprocessing
3. Feature engineering and transformation
4. Model training and evaluation
5. Model deployment readiness
6. CI/CD automation for continuous integration and updates

## 📦 Dataset Description

The dataset consists of **customer demographics and interaction attributes** used to predict the likelihood of purchasing the Wellness Tourism Package.

### Target Variable

* **ProdTaken**: Indicates whether the customer purchased the package

  * `0` – No
  * `1` – Yes

---

### Customer Details

* **CustomerID**: Unique identifier for each customer
* **Age**: Age of the customer
* **TypeofContact**: Mode of contact (Company Invited / Self Inquiry)
* **CityTier**: City category based on development and population (Tier 1 > Tier 2 > Tier 3)
* **Occupation**: Customer’s profession (Salaried, Freelancer, etc.)
* **Gender**: Gender of the customer (Male, Female)
* **NumberOfPersonVisiting**: Total number of people traveling with the customer
* **PreferredPropertyStar**: Preferred hotel star rating
* **MaritalStatus**: Marital status (Single, Married, Divorced)
* **NumberOfTrips**: Average number of annual trips
* **Passport**: Passport availability (0: No, 1: Yes)
* **OwnCar**: Car ownership (0: No, 1: Yes)
* **NumberOfChildrenVisiting**: Number of children below 5 years
* **Designation**: Job designation
* **MonthlyIncome**: Monthly income of the customer

---

### Customer Interaction Data

* **PitchSatisfactionScore**: Customer satisfaction score for the sales pitch
* **ProductPitched**: Product pitched to the customer
* **NumberOfFollowups**: Number of follow-ups after the sales pitch
* **DurationOfPitch**: Duration (in minutes) of the sales pitch

## 📊 Machine Learning Model

The model used is a tuned XGBoost Classifier.

## 📝 Business Impact

This predictive MLOps solution enables:

* Accurate identification of high-potential customers.
* Reduced manual effort and operational inefficiencies by including batch prediction option.
* Improved marketing campaign performance by trying several scenarios.
* Scalable and reproducible model deployment
* Faster adaptation to changing customer behavior

---
