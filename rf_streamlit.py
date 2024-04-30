import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the machine learning model
model = joblib.load('RF_class_best.pkl')

def main():
    st.title('Machine Learning Model Deployment')

    # Continuous variables
    credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
    age = st.number_input('Age', min_value=18, max_value=100)
    balance = st.number_input('Balance', min_value=0.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

    # Categorical variables (with placeholders for dropdown menus)
    geography = st.selectbox('Geography', ['Spain', 'France', 'Germany'])  
    gender = st.selectbox('Gender', ['Male', 'Female'])  
    tenure = st.selectbox('Tenure', [0,1,2,3,4,5,6,7,8,9,10])  
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)
    has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No']) 
    is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])  

    if st.button('Make Prediction'):
        # Encode categorical variables
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder()
        # geography_encoded = onehot_encoder.fit_transform([geography])[0]
        geography_encoded = one_hot_encoder.transform(self.x_train[['Geography']])
        gender_encoded = label_encoder.fit_transform([gender])[0]
        # tenure_encoded = label_encoder.fit_transform([tenure])[0]
        has_cr_card_encoded = label_encoder.fit_transform([has_cr_card])[0]
        is_active_member_encoded = label_encoder.fit_transform([is_active_member])[0]

        # Create the input features array (ensure correct order); harus sama sesuai dengan urutan input model
        features = [
            credit_score, geography_encoded, gender_encoded, age, tenure_encoded, balance, 
            num_of_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary
        ]

        result = make_prediction(features)

        # Modify output based on prediction (assuming 0 is negative, 1 is positive)
        if result == 0:
            output_text = "No"  # You can customize these labels as needed
        else:
            output_text = "Yes"

        st.success(f'The prediction is: {output_text}')

def make_prediction(features):
    # Reshape the input to a 2D array for the model
    input_array = np.array(features).reshape(1, -1) # Reshape to 2D array; 1 row, -1: automatically determine the number of columns
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()