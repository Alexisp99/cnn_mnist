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
    
    fig = px.imshow(draw, color_continuous_scale= "gray")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def concat(pred):
        st.session_state["predict"] = st.session_state["predict"].append(pred)

@st.cache_resource
def load_models():
    return load_model("data/model_conv_ad_norm.h5")

def drawing() :
    
    # Create a canvas component  
    drawing_mode = st.selectbox(
        "Drawing tool:", ("freedraw", "line")
    )
    stroke_width = st.slider("Stroke width: ", 15, 20, 15)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
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
    
    return draw

def handle_prediction_data(predicted_label):
    # Initialize the DataFrame if it doesn't exist in the session state
    if 'prediction_dataframe' not in st.session_state:
        columns = ['Predicted Label', 'User Confirmation']
        st.session_state.prediction_dataframe = pd.DataFrame(columns=columns)

    # Check if we've reached the maximum number of predictions (5)
    if len(st.session_state.prediction_dataframe) >= 5:
        st.warning("You've reached the maximum number of predictions!")
        
        # Calculate the average correctness of predictions and display it
        avg_correct = st.session_state.prediction_dataframe['User Confirmation'].apply(lambda x: 1 if x == 'True' else 0).mean()
        st.write(f"Average correctness of predictions: {avg_correct * 100:.2f}%")
        return  # Exit the function since the limit has been reached

    col1, col2,col3 = st.columns([1,1,10])
    
    with col1 :
        # Buttons for user confirmation
        btn_true = st.button("True")
    with col2:
        
        btn_false = st.button("False")

    # If user presses one of the buttons, add the data to the DataFrame
    if btn_true or btn_false:
        data = {
            'Predicted Label': [predicted_label],
            'User Confirmation': ['True' if btn_true else 'False']
        }
        new_row = pd.DataFrame(data)
        st.session_state.prediction_dataframe = pd.concat([st.session_state.prediction_dataframe, new_row], ignore_index=True)
    
    
        
    # Display the accumulated data
    st.write(st.session_state.prediction_dataframe)

def reset_session_state():
    if 'prediction_dataframe' in st.session_state:
        del st.session_state.prediction_dataframe
    if 'counter' in st.session_state:
        del st.session_state.counter

    
    
st.title('Draw and let the IA guess the number !')
st.write('')   



left_bar = st.sidebar
with left_bar :

    pass



# Load the model once at the beginning
model_ad_norm = load_models()

# Create two columns for drawing and user figure
col1, col2 = st.columns(2)

# Column 1: Drawing
with col1:
    draw = drawing()
    pred_ad_n = model_ad_norm.predict(draw.reshape(1,28,28))
    pred_class_n = np.argmax(pred_ad_n, axis=1)

# Column 2: User figure
with col2:
    st.write(user_figure())
    
st.write(np.round(pred_ad_n, 3))
st.write(f"Number predict : {pred_class_n.item()}")


# Example usage within your main app:
handle_prediction_data(pred_class_n[0])
# Add a Reset button at the end
if st.button("Reset"):
    reset_session_state()
    st.experimental_rerun() 

