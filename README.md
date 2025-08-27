# PDT
Official Implementation of SIGGRAPH 2025 paper "Point Distribution Transformation with Diffusion Models"




<p align="center">
  <img src="assets/teaser_crop.gif" alt="First GIF" style="width: 80%;"/>
</p>


**[Project Page](https://shanemankiw.github.io/PDT)** | **[Paper](https://arxiv.org/abs/2507.18939)**

>Point-based representations have consistently played a vital role in geometric data structures. Most point cloud learning and processing methods typically leverage the unordered and unconstrained nature to represent the underlying geometry of 3D shapes. However, how to extract meaningful structural information from unstructured point cloud distributions and transform them into semantically meaningful point distributions remains an under-explored problem. We present PDT, a novel framework for point distribution transformation with diffusion models. Given a set of input points, PDT learns to transform the point set from its original geometric distribution into a target distribution that is semantically meaningful. Our method utilizes diffusion models with novel architecture and learning strategy, which effectively correlates the source and the target distribution through a denoising process. Through extensive experiments, we show that our method successfully transforms input point clouds into various forms of structured outputs - ranging from surface-aligned keypoints, and inner sparse joints to continuous feature lines. The results showcase our framework's ability to capture both geometric and semantic features, offering a powerful tool for various 3D geometry processing tasks where structured point distributions are desired.

![teaser](./assets/teaser.jpg)

## 🔥 News
Code partially (for joints and garment lines) released! Please try and see if they work. We are also working on releasing the remeshing code very soon.

## Usage


### Environment
```bash
conda create -n pdt python=3.9
conda activate pdt
pip install -r requirements.txt
```

### Data
You can download the checkpoints and data from [here](https://tamucs-my.sharepoint.com/:f:/g/personal/jionghao_tamu_edu/Enm1rQd2EURLnwbSm9sKQ5EBz70akS4nSliNapTw445JUA?e=hRNXNK), put data/ and checkpoints/ under the main folder.

We use [RigNet](https://github.com/zhan-xu/RigNet) dataset and [GarmentCode](https://github.com/maria-korosteleva/GarmentCode) dataset. Please cite them if you are using our processed version:
```bibtex
@inproceedings{GarmentCodeData:2024,
  author = {Korosteleva, Maria and Kesdogan, Timur Levent and Kemper, Fabian and Wenninger, Stephan and Koller, Jasmin and Zhang, Yuhan and Botsch, Mario and Sorkine-Hornung, Olga},
  title = {{GarmentCodeData}: A Dataset of 3{D} Made-to-Measure Garments With Sewing Patterns},
  booktitle={Computer Vision -- ECCV 2024},
  year = {2024},
  keywords = {sewing patterns, garment reconstruction, dataset},
}

@article{GarmentCode2023,
  author = {Korosteleva, Maria and Sorkine-Hornung, Olga},
  title = {{GarmentCode}: Programming Parametric Sewing Patterns},
  year = {2023},
  issue_date = {December 2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {42},
  number = {6},
  doi = {10.1145/3618351},
  journal = {ACM Transaction on Graphics},
  note = {SIGGRAPH ASIA 2023 issue},
  numpages = {16},
  keywords = {sewing patterns, garment modeling}
}

@InProceedings{AnimSkelVolNet,
  title={Predicting Animation Skeletons for 3D Articulated Models via Volumetric Nets},
  author={Zhan Xu and Yang Zhou and Evangelos Kalogerakis and Karan Singh},
  booktitle={2019 International Conference on 3D Vision (3DV)},
  year={2019}
}

@article{RigNet,
  title={RigNet: Neural Rigging for Articulated Characters},
  author={Zhan Xu and Yang Zhou and Evangelos Kalogerakis and Chris Landreth and Karan Singh},
  journal={ACM Trans. on Graphics},
  year={2020},
  volume={39}
}
```


### Joint Prediction

To run inference:
```bash
bash scripts/infer_rignet.sh
```

To train the model, please run:
```bash
bash scripts/train_rignet.sh
```

### Garment Feature Line

To run inference:
```bash
bash scripts/infer_garcode.sh
```

To train the model, please run:
```bash
bash scripts/train_garcode.sh
```

### Remesh
Coming soon...


## Applications

<p align="center">
  <img src="assets/PDT_remeshing.gif" alt="First GIF" style="width: 80%;"/>
</p>
<p align="center"><em>Results & denoising processes of mesh keypoints prediction.</em></p>

<p align="center">
  <img src="assets/PDT_joints.gif" alt="First GIF" style="width: 80%;"/>
</p>
<p align="center"><em>Results & denoising processes of skeletal joints prediction.</em></p>

<p align="center">
  <img src="assets/PDT_garment.gif" alt="First GIF" style="width: 80%;"/>
</p>
<p align="center"><em>Results & denoising processes of feature line extraction.</em></p>


## Core Idea

<p align="center">
  <img src="assets/PDT_process.gif" alt="Process" style="width: 80%;"/>
</p>
<p align="center"><em>We pair noisy points from Gaussian distribution each with an input point as a per-point reference. Then, our diffusion model is trained to drag and denoise the Gaussian noise into a desired structural points distribution.</em></p>

## Acknowledgement
We build our main frame after [DiT-3D](https://github.com/DiT-3D/DiT-3D) and [PVD](https://github.com/alexzhou907/PVD). We also borrow a lot of data functions from [MeshGPT](https://github.com/audi/MeshGPT). Really appreciate them for open sourcing!

## Citation

If you find this work helpful, please consider cite our paper:
```bibtex
@inproceedings{10.1145/3721238.3730717,
author = {Wang, Jionghao and Lin, Cheng and Liu, Yuan and Xu, Rui and Dou, Zhiyang and Long, Xiaoxiao and Guo, Haoxiang and Komura, Taku and Wang, Wenping and Li, Xin},
title = {PDT: Point Distribution Transformation with Diffusion Models},
year = {2025},
isbn = {9798400715402},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3721238.3730717},
doi = {10.1145/3721238.3730717},
series = {SIGGRAPH Conference Papers '25}
}
```

