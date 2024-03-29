import numpy as np
from io import BytesIO
import os
import json
import requests
#import urllib.request
from stqdm import stqdm
from PIL import Image
import pickle
import streamlit as st
import sys
import urllib
import torch
import random
import biggan
from torchvision.utils import make_grid
from io import BytesIO
import base64

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    first_run = not os.path.exists('state.json') 
    state = {}
    st.title("Microstructure GAN demo")
    """This is a demonstration of conditional image generation of micrographs using a modified architecture based on [BigGAN-deep](https://arxiv.org/abs/1809.11096)
       The images generated are using three conditional inputs Annealing Temperature, Annealing Time and the type of cooling used.
       GAN is trained using [Omni Loss](https://arxiv.org/abs/2011.13074) on [UHCSDB](http://uhcsdb.materials.cmu.edu/) images. Details on the methodology can be found in the [paper](https://arxiv.org/abs/2107.09402)"""
    
    st.sidebar.title('Processing Conditions',)
    state['anneal_temp'] = st.sidebar.selectbox('Annealing Temperature °C',[700,750,800,900,970,1000,1100])
    state['anneal_time'] = st.sidebar.selectbox('Annealing Time (M: Minutes, H: Hours)',['5M','90M','1H','3H','8H','24H','48H','85H'])
    state['cooling'] = st.sidebar.selectbox('Cooling Type',['Quench','Furnace Cool','Air Cool','650C-1H'])
    temp_dict = {970: 0, 800: 1, 900: 2, 1100: 3, 1000: 4, 700: 5, 750: 6}
    time_dict = {'90M': 0, '24H': 1, '3H': 2, '5M': 3, '8H': 4, '85H': 5, '1H': 6, '48H': 7}
    cool_dict = {'Quench': 0, 'Air Cool': 1, 'Furnace Cool': 2, '650C-1H': 3}
    model = load_gan()
    st.sidebar.subheader('Generate a new latent Vector')
    state['seed'] =  7
    if st.sidebar.button('New z'):
        state['seed'] = random.randint(0,1000)
    rng = np.random.RandomState(state['seed'])
    noise = torch.tensor(rng.normal(0, 1, (1, 384))).float()
    state['noise'] = noise.numpy()
    y_temp = temp_dict[state['anneal_temp']]
    y_time = time_dict[state['anneal_time']]
    y_cool = cool_dict[state['cooling']]

    state['image_out'] = generate_img(model, noise, y_temp, y_time, y_cool)
    st.subheader('Generated Microstructure for the given processing conditions')
    st.text("")
    st.text(f"Random seed: {state['seed']}")
    st.image(np.array(state['image_out']), use_column_width=False)
    im = Image.fromarray((np.array(state['image_out']).reshape(256,256) * 255).astype(np.uint8))
    buf =  BytesIO()
    im.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label = "Download Micrograph", data = byte_im , file_name = f"{state['anneal_temp']}-{state['anneal_time']}-{state['cooling']}-{state['seed']}.png", mime="image/png")
 
    with open('state.json', 'w') as fp:
        json.dump(state, fp, cls=NumpyEncoder)
        
def download_model(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    with st.spinner(text= f"Downloading {filename} ..."):
        chunkSize = 1024
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            pbar = stqdm( unit="B", total=int( r.headers['Content-Length'] ) )
            for chunk in r.iter_content(chunk_size=chunkSize): 
                if chunk: # filter out keep-alive new chunks
                    pbar.update (len(chunk))
                    f.write(chunk)

@st.cache(suppress_st_warning=True)
def load_gan():
    if not os.path.isfile("BigGAN-deep.pth"):
        url = "https://github.com/safi842/Microstructure-GAN/releases/download/v0/BigGAN-deep.pth"
        filename = "BigGAN-deep.pth"
        download_model(url, filename)
        st.write(f"Downloaded {filename}...")
        #filename, headers = urllib.request.urlretrieve(url,"BigGAN-deep.pth",MyProgressBar())
    model = biggan.Generator()
    model.load_state_dict(torch.load('BigGAN-deep.pth', map_location=torch.device('cpu')))
    return model

@st.cache(suppress_st_warning=True)
def generate_img(model,noise, y_temp, y_time, y_cool):
    y_temp = torch.tensor([y_temp])
    y_time = torch.tensor([y_time])
    y_cool = torch.tensor([y_cool])
    with torch.no_grad():
        synthetic = model(noise, y_temp, y_time, y_cool)[0]
        synthetic = 0.5 * synthetic + 0.5
    #synthetic = make_grid(synthetic, normalize=True)
    return np.transpose(synthetic.numpy() ,(1,2,0))

main()
st.markdown('<div style="text-align: center;">Copyright (c) 2023 Mohammad Safiuddin</div>', unsafe_allow_html=True)