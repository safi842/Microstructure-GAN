# Microstructure-GAN
Conditional image generation of micrographs using Microstructure GAN which is based on [BigGAN-deep architecture](https://arxiv.org/abs/1809.11096)
The images are generated using three conditional inputs Annealing Temperature, Annealing Time and the type of cooling used.
GAN is trained using [Omni Loss](https://arxiv.org/abs/2011.13074) on [UHCSDB](http://uhcsdb.materials.cmu.edu/) images. Additionally a [Streamlit](https://www.streamlit.io/) dashboard is included for demonstration and inference.

