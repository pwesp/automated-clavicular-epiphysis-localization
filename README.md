# !Repository Under Development!

<p align="center">
  <img src="https://user-images.githubusercontent.com/56682642/156801158-44567c7d-85e0-4b08-ab52-2b01e710fa06.jpg" />
</p>

## automated-clavicular-epiphysis-localization

This repository contains Python scripts and Jupyter notebooks to localize the medial clavicular epiphyseal cartilage in CT scans, using deep learning-based object detection.

A description of how to use the code in this repository can be found in the sections below. For demonstration and test purposes, exemplaric training and test data will be provided in the folder 'data' soon.

A trained instance of the object detection network RetinaNet with a ResNet18 backbone is provided in 'model_weights.pt'.

At the core of this code is a Pytorch implementation of the RetinaNet which is based on the following GitHub repository: https://github.com/yhenon/pytorch-retinanet

### 0. Prerequisits

### 1. Training the object detection network RetinaNet

### 2. Estimating the location of the structure-of-interest (SOI) using the trained RetinaNet
