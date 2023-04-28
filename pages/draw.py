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
def data():
    return pd.read_csv("data/test.csv")

def user_figure() :
    
    fig = px.imshow(draw)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@st.cache_resource
def load_models():
    return load_model("data/model_conv_ad_norm.h5")



left_bar = st.sidebar
with left_bar :

    
    # Create a canvas component  
    drawing_mode = st.selectbox(
        "Drawing tool:", ("freedraw", "line")
    )
    stroke_width = st.slider("Stroke width: ", 15, 20, 15)
    
    canvas_result = st_canvas(
        fill_color="",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        #background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit= False,
        height=168,
        width= 168,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    
    # st.image(canvas_result.image_data)
    draw = canvas_result.image_data[::6,::6][:,:,3]/255
    
    
    

st.title('Draw and let the IA guess the number !')
st.write('')




model_ad_norm = load_models()
pred_ad_n = model_ad_norm.predict(draw.reshape(1,28,28))
pred_class_n = np.argmax(pred_ad_n,axis=1)


st.write(user_figure())
st.write(np.round(pred_ad_n,3))
st.write(pred_class_n)

