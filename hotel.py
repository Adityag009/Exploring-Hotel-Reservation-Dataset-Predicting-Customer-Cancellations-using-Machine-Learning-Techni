import streamlit as st
import pickle
import numpy as np

# Load your trained model
def load_model():
    with open('finalized_model1.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title of your app
st.title("Hotel Booking Prediction")

# Create input fields for all the features your model requires

# Numeric inputs
no_of_adults = st.number_input("Number of Adults", min_value=0, max_value=10, step=1)
no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
no_of_weekend_nights = st.number_input("Number of weekend nights", min_value=0, max_value=7, step=1)
no_of_week_nights = st.number_input("Number of week nights", min_value=0, max_value=17, step=1)
lead_time = st.number_input("Lead Time (Number of days between booking and arrival)", min_value=0, max_value=500, value=0, step=1)
avg_price_per_room = st.number_input("Average Price Per Room (in euros)", min_value=0.0, value=100.0, format="%.2f")
no_of_previous_cancellations = st.number_input("Number of previous cancellations", min_value=0, max_value=15, step=1)
no_of_special_requests = st.number_input("Number of special requests", min_value=0, max_value=5, step=1)
arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, step=1)
arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, step=1)

# Select and Radio inputs
required_car_parking_space = st.selectbox("Does the customer require a car parking space?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
repeated_guest = st.radio("Is the customer a repeated guest?", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Meal Plan Select Box
selected_meal_plan = st.selectbox("Type of Meal Plan", ("Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"))
meal_plan_encoded = {"Meal Plan 1": 0, "Meal Plan 2": 0, "Meal Plan 3": 0, "Not Selected": 0}
meal_plan_encoded[selected_meal_plan] = 1
meal_plan_features = list(meal_plan_encoded.values())

# Room Type Select Box
selected_room_type = st.selectbox("Reserved Room Type", ("Reserved Room Type 1", "Reserved Room Type 2", "Reserved Room Type 3", "Reserved Room Type 4", "Reserved Room Type 5", "Reserved Room Type 6", "Reserved Room Type 7"))
room_type_encoded = {"Reserved Room Type 1": 0, "Reserved Room Type 2": 0, "Reserved Room Type 3": 0, "Reserved Room Type 4": 0, "Reserved Room Type 5": 0, "Reserved Room Type 6": 0, "Reserved Room Type 7": 0}
room_type_encoded[selected_room_type] = 1
reserved_room_type = list(room_type_encoded.values())

# Market Segment Type Select Box
market_segment_type = st.selectbox("Market segment type", ("Aviation", "Complementary", "Corporate", "Offline", "Online"))
market_segment_encoded = {"Aviation": 0, "Complementary": 0, "Corporate": 0, "Offline": 0, "Online": 0}
market_segment_encoded[market_segment_type] = 1
market_segment_features = list(market_segment_encoded.values())

# Button to Predict
if st.button("Predict Booking Status"):
    # Prepare the feature array for prediction
    features = np.array([[no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, lead_time, avg_price_per_room, no_of_previous_cancellations, no_of_special_requests, required_car_parking_space, repeated_guest,arrival_month,arrival_date] + meal_plan_features + reserved_room_type + market_segment_features])
    
    # Make prediction
    prediction = model.predict(features)

    # Display the prediction
    if prediction[0] == 1:
        st.success("The booking is likely to be cancelled.")
    else:
        st.success("The booking is likely to be maintained.")


