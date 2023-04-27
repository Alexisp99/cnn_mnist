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

left_bar = st.sidebar
# Specify canvas parameters in application

with left_bar :
    
    # Create a canvas component  
    drawing_mode = st.selectbox(
        "Drawing tool:", ("freedraw", "line")
    )

    stroke_width = st.slider("Stroke width: ", 12, 20, 12)
    
    bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.checkbox("Update in realtime", False)
    
    canvas_result = st_canvas(
        fill_color="",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=168,
        width= 168,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    
    # st.image(canvas_result.image_data)
    draw = canvas_result.image_data[::6,::6][:,:,3]/255
    st.write(draw)
    
    

st.title('Estimate price of your future house !')
st.write('')


st.sidebar.header('Select Dataset')

col1,col2 = st.columns(2)

with col1 :
    uploaded_file = st.file_uploader("Choose a file")
try :
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
except :
    pass


test = pd.read_csv("data/test.csv")



st.cache()
baseline_model= load_model("data/model_baseline.h5")
pred = baseline_model.predict(draw.reshape(1,784))
pred_classes = np.argmax(pred,axis = 1)
st.write(px.imshow(draw))
st.write(pred)
st.write(pred_classes)

ad_conv_model = load_model("data/model_conv_ad.h5")
pred_ad = ad_conv_model.predict(draw.reshape(1,28,28))
pred_class = np.argmax(pred_ad,axis=1)
st.write(pred_ad)
st.write(pred_class)

ad_conv_norm_model = load_model("data/model_conv_ad_norm.h5")
pred_ad_n = ad_conv_norm_model.predict(draw.reshape(1,28,28))
pred_class_n = np.argmax(pred_ad_n,axis=1)
st.write(np.round(pred_ad_n,3))
st.write(pred_ad_n.round(3))
st.write(pred_class_n)

image = Image.open("chiffre/chiffre5.png")
image = image.resize((28,28))
st.image(image)
st.write(np.asarray(image)[:,:,3])

st.write(ad_conv_norm_model.predict(np.asarray(image)[:,:,3].reshape(1,28,28)))