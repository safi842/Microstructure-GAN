## Microstructure-GAN &mdash; Pytorch Implementation 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://safi842-microstructure-gan-appgan-app-32c049.streamlit.app/)


![Overview](https://github.com/safi842/Microstructure-GAN/blob/main/docs/Omni%20BigGAN%20-%20Overview.jpg)

### Establishing process-structure linkages using Generative Adversarial Networks<br>
Mohammad Safiuddin, Ch Likith Reddy, Ganesh Vasantada, CHJNS Harsha, Dr. Srinu Gangolu<br>

Paper: https://arxiv.org/abs/2107.09402<br>

[comment]: <> (Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila<br>)
[comment]: <> (Paper: http://arxiv.org/abs/1912.04958<br>)

Abstract: *The microstructure of a material strongly influences its mechanical properties
and the microstructure itself is influenced by the processing conditions. Thus,
establishing a Process-Structure-Property relationship is a crucial task in material design and is of interest in many engineering applications. In this work,
the processing-structure relationship is modelled as deep learning based conditional image synthesis problem. This approach is devoid of feature engineering,
needs little domain awareness, and can be applied to a wide variety of material
systems. We develop a GAN (Generative Adversarial Network) to synthesize
microstructures based on given processing conditions. Results show that our GAN model
can produce high-fidelity multiphase microstructures which have a good correlation with the given processing conditions.*

### Results:

<p align="center">
  <img src="https://github.com/safi842/Microstructure-GAN/blob/main/docs/Gen%20vs%20Real.jpg" width="500" />
</p>

#### File Overview

The following files are included in this package:

- `omni-loss-biggan.ipynb`: an Ipython notebook that contains the code used to train the model.
- `new_metadata.xlsx`: an Excel workbook that holds the training image metadata.
- `.\app`: a directory that contains the source code for the app. Further instructions on the app can be found below.

### Application 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://safi842-microstructure-gan-appgan-app-32c049.streamlit.app/) \
**If you want to run the app locally, follow the instructions below** 
<p align="center">
  <img src="https://raw.githubusercontent.com/safi842/Microstructure-GAN/main/docs/GAN%20App%20demo.png" width="1500"/>
</p>

To install the app, unzip the `.\Microstructure_GAN` folder. Next, navigate to the `.\Microstructure_GAN\app` directory in a terminal and run the following command to install the necessary packages:

```
pip install requirements.txt
```

Once the packages have been installed, run the following command to start the web app:

```
streamlit run gan_app.py
```

**Recreating Results:**

Generated micrographs can be downloaded by clicking the "Download Micrograph" button. The file name of the saved image contains the processing conditions and seed value, for example: `800-85H-Quench-864.png`. To recreate the image, the latent vector can be generated using the `seed` as follows.
```
seed = 864
rng = np.random.RandomState(seed)
latent_vector = rng.normal(0, 1, (1, 384))
```
