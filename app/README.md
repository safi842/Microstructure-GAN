## Notes
The `requirements.txt` file should list all Python libraries that the app depends on , and they will be installed using:

```
pip install -r requirements.txt
```
Once the requirements are installed,
go to [releases](https://github.com/safi842/Microstructure-GAN/releases) and download the source code and the `BigGAN-deep.pth` file (contains the weights of the trained generator). Then extract the app directory. 
Once the app directory is extracted put the `BigGAN-deep.pth` in the same directory. The application can be started using 
```
streamlit run gan_app.py
```
### App 
Micrographs can be saved using the *Save Image* button. They can be found in `app\Generated Micrographs`. The saved image's filename contains the processing conditions and the seed value. For example: `800-85H-Quench-864.png`. The latent vector used to generate the particular image can be generated using the `seed` as follows.
```
seed = 864
rng = np.random.RandomState(seed)
latent_vector = rng.normal(0, 1, (1, 384))
```
A GPU is not required to run this app.
<p align="center">
  <img src="https://github.com/safi842/Microstructure-GAN/blob/0e8655e5c6db3e0bf0cd47461ee084d42cf06269/app/GAN%20App%20demo.png" width="1001" />
</p>
