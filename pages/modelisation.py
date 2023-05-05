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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomTranslation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.losses import sparse_categorical_crossentropy

from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

import pickle

#create the page of the web app
st.set_page_config(page_title = "Immobilier",
                   layout = "wide", 
                   )

model = load_model("data/model_conv_ad_norm.h5")
X_train = pd.read_csv("data/test.csv")

# Create a model with all layers
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Choose an image to visualize the feature maps
img_tensor = X_train.loc[4].values.reshape(1, 28, 28, 1)
st.write(px.imshow(img_tensor.reshape(28,28), color_continuous_scale= 'gray'))

# Get the activations for all layers
activations = activation_model.predict(img_tensor)



# Visualize feature maps for each layer
for layer, activation in zip(model.layers, activations):
    if isinstance(layer, Conv2D):
        filters = layer.filters
        fig = make_subplots(rows=int(filters/4), cols=4)
        for i in range(filters):
            fig.add_trace(px.imshow(activation[0, :, :, i]).data[0], row=int(i/4)+1, col=(i%4)+1)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_coloraxes(colorscale = "gray")
        fig.update_layout(title=f"Conv2D Layer {layer.name}",coloraxis_showscale=False)
        fig.update_layout(width = 1500, height = 1500)
        st.write(fig)
    elif isinstance(layer, MaxPooling2D):
        fig = make_subplots(rows=1, cols=4)
        for i in range(layer.pool_size[1]):
            fig.add_trace(px.imshow(activation[0, :, :, i]).data[0], row=1, col=i+1)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_coloraxes(colorscale = "gray")
        fig.update_layout(title=f"MaxPooling2D Layer {layer.name}",coloraxis_showscale=False)
        st.write(fig)
    elif isinstance(layer, Dense):
        fig = px.imshow(activation, color_continuous_scale='gray', width= 1500, height= 500)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(title=f"Dense Layer {layer.name}",coloraxis_showscale=False)
        st.write(fig)