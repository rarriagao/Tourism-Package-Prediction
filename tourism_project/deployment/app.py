import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction")
st.write("Choose an individual prediction or upload a CSV for batch predictions.")

classification_threshold = 0.45

# Individual prediction controls
st.header("Individual Prediction")

Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips", 1, 20, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 5, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

individual_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict Individual Customer", key="individual_predict"):
    probability = model.predict_proba(individual_data)[0, 1]
    prediction = int(probability >= classification_threshold)
    result = "will purchase the travel package" if prediction else "is unlikely to purchase"
    st.success(f"Prediction: Customer {result}")
    st.write(f"Purchase probability: {probability:.1%}")

st.divider()

# Batch prediction controls remain visible independently of the individual button
st.header("Batch Prediction")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing customer records",
    type=["csv"],
    key="batch_file",
)

if st.button("Run Batch Prediction", key="batch_predict"):
    if uploaded_file is None:
        st.warning("Upload a CSV file before running batch prediction.")
    else:
        batch_data = pd.read_csv(uploaded_file)

        if "CustomerID" in batch_data.columns:
            batch_data = batch_data.drop(columns=["CustomerID"])
        if "ProdTaken" in batch_data.columns:
            batch_data = batch_data.drop(columns=["ProdTaken"])

        required_columns = list(individual_data.columns)
        missing_columns = [
            column for column in required_columns
            if column not in batch_data.columns
        ]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
        else:
            batch_data = batch_data[required_columns]
            probabilities = model.predict_proba(batch_data)[:, 1]
            predictions = (probabilities >= classification_threshold).astype(int)

            results = batch_data.copy()
            results["PurchaseProbability"] = probabilities
            results["Prediction"] = predictions
            results["Result"] = results["Prediction"].map({
                1: "Will purchase",
                0: "Unlikely to purchase",
            })

            st.dataframe(results)
            st.download_button(
                "Download Predictions",
                results.to_csv(index=False),
                "tourism_batch_predictions.csv",
                "text/csv",
                key="download_batch_predictions",
            )
