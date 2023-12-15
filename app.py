import json
import requests
import streamlit as st
import scipy.stats as sp
import joblib
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split as train_test_split
import streamlit.components.v1 as html
from  PIL import Image
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from textblob import TextBlob

# Import EDA Packages
import pandas as pd
import numpy as np

# Import Visualisation Packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
from matplotlib import style

# Hide Warnings
import warnings
warnings.filterwarnings('ignore')

# Enable displaying all columns and preventing truncation
pd.set_option('display.max.columns', None)
pd.set_option('display.max_colwidth', None)

# Load the Text Cleaning Package
import neattext.functions as nfx

st.set_page_config(
    layout="wide",
    page_title="Twitter Sentiment Analysis",
    page_icon="C:\\Users\\tarac\\Downloads\\Cancer Death Rate Prediction - Regression Analysis\\streamlit_application\\images\\pink-ribbon.png",
    initial_sidebar_state="expanded",
)

# Hide the Streamlit Footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the Lottie File
@st.cache
def load_lottiefile(filepath: str):
    with open(filepath) as f:
        return json.load(f)

# Load lottie files
from streamlit_lottie import st_lottie # pip install streamlit-lottie
# Define the paths of the lottie files
path = "./assets/lottie files/twitter-lottie.json"

# Load the Lottie File
lottie_twitter = load_lottiefile(path)

# Display the Lottie File in the sidebar
with st.sidebar:
    st_lottie(lottie_twitter, quality='high', speed=1, height=300, key="initial")

# Load the Models
log_clf_CV = joblib.load('models/log_clf_CV.pkl')

# Define the paths of the lottie files
lottie_files = {
    "Positive" : "./assets/lottie files/Positive - 1701447921922.json",
    "Neutral" : "./assets/lottie files/Neutral - 1701448191091.json",
    "Negative" : "./assets/lottie files/Negative - 1701448132377.json",
}

# Fxn
def predict_sentiment(docx):
    results = log_clf_CV.predict([docx])
    return results[0]

def get_lottie_file(sentiment):
    with open(lottie_files[sentiment]) as file:
        lottie_file = json.load(file)
    return lottie_file

# Link to the Roboto font
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">', unsafe_allow_html=True)

html_temp = """
    ##### <div style="background: linear-gradient(to right, #323232, #122f69); color:white; display:fill; border-radius:38px; letter-spacing:1.0px; font-family: 'Roboto', sans-serif;"><p style="padding: 52px;color:white;font-size:250%;"><b><b><span style='color:white'><span style='color:#F1A424'>|</span></span></b> Predicting Sentiment</b></p></div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

def main():
    st.subheader("Sentiment in Text")
    st.sidebar.title("Twitter Sentiment Analysis")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        col1, col2 = st.columns(2)

        # Apply Fxn here
        prediction = predict_sentiment(raw_text)

        with col1:
            st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">', unsafe_allow_html=True)
            st.markdown("""
            <h4 style='font-family: Jua'>Original Text:
            </p>
            """
            , unsafe_allow_html=True)
            st.success(raw_text)

            st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">', unsafe_allow_html=True)
            st.markdown("""
            <h4 style='font-family: Jua'>Prediction: 
            </p>
            """
            , unsafe_allow_html=True) 
            st.success(prediction)

        with col2:
            lottie_file = get_lottie_file(prediction)
            st_lottie(lottie_file, width=350, height=350)

if __name__ == '__main__':
    main()

st.sidebar.markdown('''
---
Created with ❤️ by Group 1.
''')
