import streamlit as st
import numpy as np
import joblib

# Load the simpler model
@st.cache_resource
def load_model():
    return joblib.load("rainfall_simple_model.pkl")

model = load_model()

# Page settings
st.set_page_config(
    page_title="Rainfall Predictor 🌧️",
    page_icon="🌦️",
    layout="centered"
)

# Title
st.markdown(
    "<h1 style='text-align: center; color: #3C87C7;'>🌦️ Rainfall Prediction System</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Predict if rainfall is likely today using weather measurements.</p>", 
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1779/1779940.png", width=150)
st.sidebar.title("About This App")
st.sidebar.markdown("""
**How it works:**  
This app uses a Random Forest model trained on:
- Temperature
- Humidity
- Pressure
- Wind Speed
- Wind Bearing
- Visibility

to predict whether it will rain today.  
Feel free to experiment with values! 🌈
""")

# Weather inputs
st.markdown("### 🌤️ Weather Measurements")

with st.form("rainfall_form"):
    col1, col2 = st.columns(2)

    with col1:
        avg_temp = st.slider("Average Temperature (°C)", -20.0, 50.0, 25.0, step=0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, 1013.0, step=0.5)
    
    with col2:
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 5.0, step=0.5)
        wind_bearing = st.slider("Wind Bearing (°)", 0, 360, 180)
        visibility = st.slider("Visibility (km)", 0.0, 50.0, 10.0, step=0.5)

    submitted = st.form_submit_button("🌧️ Predict")

if submitted:
    features = np.array([[avg_temp, humidity, pressure, wind_speed, wind_bearing, visibility]])
    prediction = model.predict(features)[0]
    
    st.markdown("---")

    if prediction == 1:
        st.success("🌧️ **Prediction: Rain is likely today!** Stay safe and carry an umbrella. ☔")
    else:
        st.info("☀️ **Prediction: No rain expected today.** Enjoy the sunshine! 😎")

    st.markdown(
        f"""
        #### 📊 Details:
        - **Temperature:** {avg_temp} °C  
        - **Humidity:** {humidity} %  
        - **Pressure:** {pressure} hPa  
        - **Wind Speed:** {wind_speed} km/h  
        - **Wind Bearing:** {wind_bearing} °  
        - **Visibility:** {visibility} km  
        """, 
        unsafe_allow_html=True
    )

st.markdown("""
---
<div style='text-align: center; color: gray;'>🌈 Built with Streamlit & Random Forest | by Malvika</div>
""", unsafe_allow_html=True)
