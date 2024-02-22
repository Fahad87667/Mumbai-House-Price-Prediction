import streamlit as st
import pandas as pd
import pickle

# Load data and model
data = pd.read_csv('./cleaned_data.csv')
pipe = pickle.load(open("model.pkl", "rb"))

# Define Streamlit app with a customized color theme and layout
def main():
    st.title("MumbaiDreamHomes Predictor")

    # Custom inline CSS for a different color theme
    st.markdown(
        """
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f0f0f0;
                color: #333;
            }
            .stButton button {
                background-color: #4CAF50; /* Green */
                color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 10px 20px;
                margin-top: 15px;
            }
            .stButton button:hover {
                background-color: #45a049; /* Darker Green */
            }
            .stSelectbox, .stNumberInput {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 10px;
                margin-top: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input fields under the main title
    st.text("Customize Your Dream Property")
    bhk = st.selectbox("Select BHK", sorted(data['bhk'].unique()))
    house_type = st.selectbox("Select House Type", sorted(data['type'].unique()))
    area = st.number_input("Enter Area", min_value=0)
    region = st.selectbox("Select Region", sorted(data['region'].unique()))
    status = st.selectbox("Select Status", sorted(data['status'].unique()))

    # Prediction button
    if st.button("Predict"):
        input_data = pd.DataFrame([[bhk, house_type, area, region, status]],
                                  columns=['bhk', 'type', 'area', 'region', 'status'])
        prediction = pipe.predict(input_data)[0]

        # Format prediction as a string with currency symbol
        formatted_prediction = f'₹ {prediction:.2f} Lacs'

        # Display input parameters and prediction result in the sidebar
        st.sidebar.subheader("Dream Home Setup:")
        st.sidebar.write(f"- BHK: {bhk}")
        st.sidebar.write(f"- House Type: {house_type}")
        st.sidebar.write(f"- Area: {area}")
        st.sidebar.write(f"- Region: {region}")
        st.sidebar.write(f"- Status: {status}")

        st.sidebar.subheader("Prediction Result:")
        st.sidebar.success(f"Estimated Price: {formatted_prediction}")

if __name__ == '__main__':
    main()
