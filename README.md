# Fusion Model for Histology with GNNs and CNNs

This is an implementation of our fusion model that enhances global image-level representations captured by CNNs with the geometry of cell-level spatial information learned by GNNs. This is part of the work published in the paper "[How GNNs Facilitate CNNs in Mining Geometric Information from Large-Scale Medical Images](https://arxiv.org/abs/2206.07599)".

### 1. Overview
For histology, Gigapixel medical images provide massive data to be mined. This includes morphological textures and spatial information. Our solution optimally integrates features from global images and cell graphs through a fusion layer, improving over solutions that heavily rely on CNNs for global pixel-level analysis. This solution allows for a more comprehensive representation in downstream oncology tasks such as biomarker predictions.

### 2. Dataset
#### Images are avaiable at [Link](https://zenodo.org/record/2530835#.YrLH-S-KFtQ)
#### Graph data is available at [Link](https://zenodo.org/record/6683652#.YrLjLC-KFtQ)

### 3. Code Structure
Each section of our code serves a specific purpose:
- [`mm_model.py`](mm_model.py): Model construction
- [`mm_trainer.py`](mm_trainer.py): Training code
- [`mm_evaluater.py`](mm_evaluater.py): Evaluation codes for both patch and WSI levels
- [`mm_dataset.py`](mm_dataset.py): Dataset loader
- [`main.py`](main.py): main functions

The example configuration files are in the `config/` directory for CRC and STAD datasets. Adjust the yaml file when running the codes on your devices.

To run the training script [`main.py`](main.py):
```bash
python main.py --config_path PATH_TO_CONFIG
```

To run the evaluation script [`mm_evaluater.py`](mm_evaluater.py):
```bash
python mm_evaluater.py --gpu_id 0 --path PATH_TO_THE_OUTPUT --choice acc
```

#### Contributions
This repository is maintained by Cosmosec. If you find our paper, code helpful to your research. Please consider citing our paper: 
```
@article{shen2022gnns,
  title={How GNNs Facilitate CNNs in Mining Geometric Information from Large-Scale Medical Images},
  author={Shen, Yiqing and Zhou, Bingxin and Xiong, Xinye and Gao, Ruitian and Wang, Yu Guang},
  journal={arXiv preprint arXiv:2206.07599},
  year={2022}
}
```
