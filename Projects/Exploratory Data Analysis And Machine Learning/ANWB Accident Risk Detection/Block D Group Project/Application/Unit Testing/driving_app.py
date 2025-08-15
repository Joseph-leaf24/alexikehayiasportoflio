import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

class DrivingRiskApp:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = ['latitude', 'longitude', 'speed_kmh', 'maxwaarde', 'accident_count']
        self.risk_mapping = {
            0: 'Low risk',
            1: 'Minor accident risk',
            2: 'Moderate accident risk',
            3: 'Severe accident risk'
        }
        self.setup_streamlit()

    def setup_streamlit(self):
        """Setup Streamlit configuration and custom CSS."""
        st.set_page_config(
            page_title="Driving Risk Prediction",
            page_icon="🚗",
            layout="centered",
            initial_sidebar_state="collapsed",
        )
        self.set_bg_hack_url()

    def set_bg_hack_url(self):
        """Set background image and custom CSS styles."""
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("https://img.freepik.com/free-photo/background-gradient-lights_23-2149304985.jpg?w=740&t=st=1718106053~exp=1718106653~hmac=4cc993d19378248f35268f89f719fce68151997fa41c36b3a25c6425ac7187e5");
                background-size: cover;
            }}
            .center-text {{
                text-align: center;
            }}
            .center-button {{
                display: flex;
                justify-content: center;
            }}
            .large-text {{
                font-size: 1.5em;
                font-weight: bold;
            }}
            .value-text {{
                font-size: 2em;
                font-weight: bold;
            }}
            .value-text-small {{
                font-size: 2em;
                font-weight: bold;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    def load_pretrained_model(self):
        """Load the pretrained model with specified architecture and weights."""
        model = Sequential([
            Dense(128, input_dim=5, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4, activation='softmax')
        ])
        model.load_weights('Improved_MLP_Model_2B.h5')
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def get_provided_dummy_data(self):
        """Generate and return a DataFrame of dummy data."""
        data = [
            # Example dummy data
            {'latitude': 51.57965, 'longitude': 4.803370, 'speed_kmh': 63.997307, 'maxwaarde': 71.150510, 'accident_count': 0},
            # Add more rows as needed
        ]
        return pd.DataFrame(data)

    def predict_risk(self, data):
        """Predict the risk level using the model and scaler."""
        X_scaled = self.scaler.transform(data)
        prediction = self.model.predict(X_scaled)
        risk_level = self.risk_mapping[np.argmax(prediction, axis=1)[0]]
        return prediction, risk_level

    def run(self):
        """Main function to run the Streamlit app."""
        st.markdown('<h1 class="center-text">🚗 Driving Risk Prediction 🚗</h1>', unsafe_allow_html=True)
        
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False

        if not st.session_state['logged_in']:
            self.login_page()
        else:
            self.main_page()

    def login_page(self):
        """Display the login page."""
        st.header("Login")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        col11, col12, col13, col14, col15 = st.columns(5)
        with col13:
            if st.button("Login"):
                st.session_state['logged_in'] = True
                st.experimental_rerun()

    def main_page(self):
        """Display the main application page."""
        if 'page' not in st.session_state:
            st.session_state['page'] = 'home'
        if 'interval' not in st.session_state:
            st.session_state['interval'] = 30  # Default interval
        if 'settings' not in st.session_state:
            st.session_state['settings'] = False

        if st.session_state['page'] == 'home':
            self.home_page()
        elif st.session_state['page'] == 'analyzing':
            self.analyzing_page()
        elif st.session_state['page'] == 'results':
            self.results_page()

    def home_page(self):
        """Display the home page."""
        st.header("Welcome to the Driving Risk Prediction App!")
        st.write("This app analyzes driving behavior data to predict the risk level of an accident. Click the 'Analyze' button to start the analysis.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col3:
            if st.button("Analyze"):
                st.session_state['page'] = 'analyzing'
                st.experimental_rerun()

        col6, col7, col8, col9, col10 = st.columns(5)
        with col8:
            if st.button("Settings"):
                st.session_state['settings'] = not st.session_state['settings']

        if st.session_state['settings']:
            interval = st.selectbox(
                "Select interval (seconds)",
                options=[30, 60, 120],
                index=[30, 60, 120].index(st.session_state['interval'])
            )
            st.session_state['interval'] = interval

    def analyzing_page(self):
        """Display the analyzing page."""
        st.markdown('<h2 class="center-text">Retrieving your vehicle data...</h2>', unsafe_allow_html=True)
        col16, col17, col18, col19, col20 = st.columns(5)
        with col18:
            with st.spinner("Analyzing..."):
                time.sleep(2)  # Simulate a loading time

        st.session_state['page'] = 'results'
        st.experimental_rerun()

    def results_page(self):
        """Display the results page."""
        if not self.model:
            self.load_pretrained_model()
        
        if 'dummy_data' not in st.session_state:
            st.session_state['dummy_data'] = self.get_provided_dummy_data()

        dummy_data = st.session_state['dummy_data'].sample(n=1).reset_index(drop=True)
        self.scaler.fit(st.session_state['dummy_data'][self.features])

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            col1_value = st.empty()
            col1_header = st.empty()
        with col2:
            col2_value = st.empty()
            col2_header = st.empty()
        with col3:
            col3_value = st.empty()
            col3_header = st.empty()
        with col4:
            col4_value = st.empty()
            col4_header = st.empty()
        with col5:
            col5_value = st.empty()
            col5_header = st.empty()

        progress_placeholders = [
            (col1_value, col1_header),
            (col2_value, col2_header),
            (col3_value, col3_header),
            (col4_value, col4_header),
            (col5_value, col5_header)
        ]

        display_columns = {
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'speed_kmh': 'Speed (km/h)',
            'maxwaarde': 'Max Value',
            'accident_count': 'Accident Count'
        }

        columns = self.features
        increments = [dummy_data[col].values[0] / 100 for col in columns]

        for i, col in enumerate(columns):
            progress_placeholders[i][1].markdown(f"<h3 class='center-text'>{display_columns[col]}</h3>", unsafe_allow_html=True)

        for step in range(101):
            for i, col in enumerate(columns):
                value = step * increments[i]
                if col == 'speed_kmh' and value >= 100:
                    progress_placeholders[i][0].markdown(f"<h1 class='center-text value-text-small'>{value:.2f}</h1>", unsafe_allow_html=True)
                else:
                    progress_placeholders[i][0].markdown(f"<h1 class='center-text value-text'>{value:.2f}</h1>", unsafe_allow_html=True)
            time.sleep(0.05)

        prediction, risk_level = self.predict_risk(dummy_data[self.features])
        st.success(f"Predicted Risk Level: {risk_level}")
        st.write("Prediction Probabilities:", prediction)

        # Debug information to understand predictions
        st.write("Input Data Used for Prediction:", dummy_data[self.features])
        st.write("Scaled Input Data:", self.scaler.transform(dummy_data[self.features]))

        time.sleep(st.session_state['interval'])
        st.session_state['page'] = 'analyzing'
        st.experimental_rerun()

if __name__ == "__main__":
    app = DrivingRiskApp()
    app.run()
