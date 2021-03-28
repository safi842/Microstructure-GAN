import numpy as np
import os
from PIL import Image
import pickle
import streamlit as st
import sys
import urllib
import torch
import random
import biggan
from torchvision.utils import make_grid

def main():
    st.title("Microstructure BigGAN-deep demo")
    """This is a demonstration of conditional image generation of micrographs using [BigGAN-deep architecture](https://arxiv.org/abs/1809.11096)
       The images generated are using three conditional inputs Annealing Temperature, Annealing Time and the type of cooling used.
       GAN is trained using [UHCSDB](http://uhcsdb.materials.cmu.edu/) images"""
    
    st.sidebar.title('Processing Conditions',)
    anneal_temp = st.sidebar.selectbox('Annealing Temperature °C',[700,750,800,900,970,1000,1100])
    anneal_time = st.sidebar.selectbox('Annealing Time (M: Minutes, H: Hours)',['5M','90M','1H','3H','8H','24H','48H','85H'])
    cooling = st.sidebar.selectbox('Cooling Type',['Quench','Furnace Cool','Air Cool','650C-1H'])
    temp_dict = {970: 0, 800: 1, 900: 2, 1100: 3, 1000: 4, 700: 5, 750: 6}
    time_dict = {'90M': 0, '24H': 1, '3H': 2, '5M': 3, '8H': 4, '85H': 5, '1H': 6, '48H': 7}
    cool_dict = {'Quench': 0, 'Air Cool': 1, 'Furnace Cool': 2, '650C-1H': 3}
    model = load_gan()
    st.sidebar.subheader('Generate a new latent Vector')
    if st.sidebar.button('New z'):
    	seed = random.randint(0,100)
    	torch.manual_seed(seed)
    else:
    	seed = 7
    	torch.manual_seed(seed)
    noise = torch.randn(1,128)
    image_out = generate_img(model, noise, temp_dict[anneal_temp], time_dict[anneal_time], cool_dict[cooling])
    st.subheader('Generated Microstructure for the given processing conditions')
    st.text("")
    st.image(image_out, use_column_width=False)
    if st.button('Save Image'):
    	if not os.path.exists('Generated Images'):
    		os.makedirs('Generated Images')
    	im = Image.fromarray((image_out * 255).astype(np.uint8))
    	im.save(f"./Generated Images/{anneal_temp}-{anneal_time}-{cooling}-{seed}.jpeg")

@st.cache(suppress_st_warning=True)
def load_gan():
    model = biggan.Generator()
    model.load_state_dict(torch.load('BigGAN-deep.pth', map_location=torch.device('cpu')))
    return model

@st.cache(suppress_st_warning=True)
def generate_img(model,noise, y_temp, y_time, y_cool):
	y_temp = torch.tensor([y_temp])
	y_time = torch.tensor([y_time])
	y_cool = torch.tensor([y_cool])
	with torch.no_grad():
		synthetic = model(noise, y_temp, y_time, y_cool)
	synthetic = make_grid(synthetic, normalize=True)
	return np.transpose(synthetic.numpy() ,(1,2,0))



if __name__ == "__main__":
  main()
