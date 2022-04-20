# Fusion Model for Histology with GNNs and CNNs

This is an implementation of our fusion model that enhances global image-level representations captured by CNNs with the geometry of cell-level spatial information learned by GNNs. This is part of the work published in the paper "[How GNNs Facilitate CNNs in Mining Geometric Information from Large-Scale Medical Images](https://arxiv.org/abs/2206.07599)".

### 1. Overview
For histology, Gigapixel medical images provide massive data to be mined. This includes morphological textures and spatial information. Our solution optimally integrates features from global images and cell graphs through a fusion layer, improving over solutions that heavily rely on CNNs for global pixel-level analysis. This solution allows for a more comprehensive representation in downstream oncology tasks such as biomarker predictions.

### 2. Dataset
#### Images are avaiable a