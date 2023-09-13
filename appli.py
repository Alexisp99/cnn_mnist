import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

#create the page of the web app
st.set_page_config(page_title = "Immobilier",
                   layout = "wide", 
                   )


@st.cache_data
def import_data(name):
    return pd.read_csv(f"data/{name}.csv")

@st.cache_resource
def load_models():
    return load_model("data/model_conv_ad_norm.h5")

def user_figure(draw) :
    
    fig = px.imshow(draw, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

# setting up the session state
if "random_num" not in st.session_state:
    st.session_state.random_num = None
    
def image_pred() : 
    # Initialize or get the test data
    test = np.asarray(import_data("test"))
    
    # Initialize session state for dataset index
    if "data_index" not in st.session_state:
        st.session_state.data_index = np.random.randint(0, test.shape[0])

    image = user_figure(test[st.session_state.data_index].reshape(28, 28))
    st.write(image)

    prediction = st.button("Predict")
    choose_number = st.button("reset")

    if choose_number:
        # Update session state with a new random index
        st.session_state.data_index = np.random.randint(0, test.shape[0])

    if prediction:
        model = load_models()
        pred = model.predict(test[st.session_state.data_index].reshape(1, 28, 28, 1))
        pred_class = np.argmax(pred, axis=1)
        st.write(np.round(pred, 3))
        st.write(f"Predicted class: {pred_class[0]}")





left_bar = st.sidebar
with left_bar :
    pass
   
    
   

st.title('Use the AI !')
st.write('')

image_pred()

