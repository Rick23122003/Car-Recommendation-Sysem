import streamlit as st

# Import project modules
from src.Data_Preprocessing import load_and_preprocess_data
from src.Recommendation import recommend_cars



# Page configuration

st.set_page_config(
    page_title="Car Recommendation System",
    page_icon="🚗",
    layout="centered"
)



# App Title & Description

st.title("Car Recommendation System")
st.write(
    """
    This application recommends similar cars based on
    **engine, mileage, power, price, fuel type, and transmission**.
    """
)



# Load Data

@st.cache_data
def load_data():
    return load_and_preprocess_data()

df = load_data()

# Extract unique car names
car_list = sorted(df["Name"].dropna().unique())



# User Input Section

st.subheader("Select a car")

selected_car = st.selectbox(
    "Choose a car from the list:",
    car_list
)

top_n = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)



# Recommendation Button

if st.button("🔍 Recommend Cars"):

    recommendations = recommend_cars(selected_car, top_n=top_n)

    if isinstance(recommendations, str):
        st.warning(recommendations)
    elif recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader("Recommended Cars")
        for i, car in enumerate(recommendations, start=1):
            st.write(f"{i}. {car}")



# Footer
st.markdown("---")
st.caption("Built with using Python, Pandas, Scikit-learn, and Streamlit")