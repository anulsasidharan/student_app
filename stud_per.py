import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://anu:tiger@cluster0.57jxgvp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']
collection = db['student_pred']


# Step 1: Load the model
def load_model():
    with open("student_lr_final_model.pkl", "rb")  as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

# Step 2: Custom function to get the data and transform into standardizeed form
def preprocessing_input_data(data, scaler, le):
    #Convert the categorical data into numeric form
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

# Step 3: Model Predection
def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocessing_input_data(data, scaler, le)
    predection = model.predict(processed_data)
    return predection

# Step 4: define the main function

def main():
    st.title("Student Performance Predection App")
    st.write("Enter your data to get a predection for your performance")

    # Create fields so that the users can insert the data
    # This data will be stored in a variable so that we can pass the data to the model. 

    hours_studied = st.number_input("Hours studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Scores", min_value=40, max_value=100, value=70)
    Extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    sleep_hours = st.number_input("Sleep Hours", min_value=4, max_value=9, value=7)
    practiced_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=10, value=5)

    # Submit button
    if st.button("Predict_your_score"):
        # Do the data mapping
        user_data = {
            "Hours Studied":hours_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":Extracurricular_activities,
            "Sleep Hours":sleep_hours,
            "Sample Question Papers Practiced":practiced_papers
        }

        prediction = predict_data(user_data)
        st.success(f"Your predection result is {prediction}")
        user_data['prediction'] = round(float(prediction[0]),2)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        collection.insert_one(user_data)


if __name__ == "__main__":
    main()
