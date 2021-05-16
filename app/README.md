## Notes
The `requirements.txt` file should list all Python libraries that the app depends on , and they will be installed using:

```
pip install -r requirements.txt
```
Once the requirements are installed,
go to [releases](https://github.com/safi842/Microstructure-GAN/releases) and download the source code and the BigGAN-deep.pth file (contains the weights of the trained generator). Then extract the app directory. 
Once the app directory is extracted put the BigGAN-deep.pth in the same directory. The app can be started using 
```
streamlit run gan_app.py
```
