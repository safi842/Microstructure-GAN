## Microstructure-GAN &mdash; Pytorch Implementation

![Overview](https://github.com/safi842/Microstructure-GAN/blob/main/docs/Omni%20BigGAN%20-%20Overview.jpg)

**Establishing process-structure linkages using Generative Adversarial Networks**<br>
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

### App 
<p align="center">
  <img src="https://raw.githubusercontent.com/safi842/Microstructure-GAN/main/docs/GAN%20App%20demo.png" width="1500"/>
</p>

Micrographs can be saved using the *Save Image* button. They can be found in `app\Generated Micrographs`. The saved image's filename contains the processing conditions and the seed value. For example: `800-85H-Quench-864.png`. The latent vector used to generate the particular image can be reproduced using the `seed` as follows.
```
seed = 864
rng = np.random.RandomState(seed)
latent_vector = rng.normal(0, 1, (1, 384))
```
## Citation
```
@misc{safiuddin2021establishing,
      title={Establishing process-structure linkages using Generative Adversarial Networks}, 
      author={Mohammad Safiuddin and CH Likith Reddy and Ganesh Vasantada and CHJNS Harsha and Srinu Gangolu},
      year={2021},
      eprint={2107.09402},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

