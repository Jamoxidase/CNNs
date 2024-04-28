# Capsule Network for MNIST Digit Classification

This notebook contains a PyTorch implementation of a Capsule Network (CapsNet) trained to classify handwritten digits from the MNIST dataset. Capsule Networks, introduced by Hinton et al. in "Dynamic Routing Between Capsules" (2017), offer an interesting alternative to traditional convolutional neural networks (CNNs) by seeking to capture hierarchical relationships between features with a parse-tree like approach designed to capture hierarchical features in high-level capsules. Laue et al. in "Why Capsule Neural Networks Do Not Scale: Challenging the Dynamic Parse-Tree Assumption" (2023) present findings that highlight some limitations of CapsNet arcitecture and a lack of scalability, although I do not think that this impedes on the interesting thought experement that is CapsNet. I had a good time learning about this arcitecture as it is based on a "natural" approach to feature extraction, although there is some residual numerical instability in my current implimentation. Nonetheless it achievs 99%+ accuracy over the full MNIST test set and is generalizible to custom images, and this was my first crack at MNIST which has been pretty cool to explore. 

## Overview

This notebook includes the following components:

- `CapsNet.py`: Contains the implementation of the Capsule Network model (`CapsNet` class) and its associated components such as convolutional layers (`ConvLayer`), primary capsules (`PrimaryCaps`), digit capsules (`DigitCaps`), and the decoder (`Decoder`).

- `train.py`: Implements the training loop for the Capsule Network model. It iterates over epochs, updates model parameters, and evaluates performance.

- `evaluate.py`: Provides utility functions for evaluating the model's performance on test data.

- `inference.ipynb`: A Jupyter notebook where users can upload their own handwritten digit images for inference using the trained Capsule Network model.

## Usage

1. **Training**: Run `train.py` to train the Capsule Network model. You can customize training parameters such as batch size, optimizer/learning rate, and number of epochs in the script.

```bash
python train.py
```

2. **Inference**: Use the `inference.ipynb` notebook to upload your own handwritten digit images and run inference using the trained model. Follow the instructions provided in the notebook for step-by-step guidance.

## Known Issues

There is a known numerical instability issue causing the model to degrade rapidly after excessive epochs. Efforts are underway to address this issue and improve model stability.

## Acknowledgments

Special thanks to Hinton et al. for their pioneering work on Capsule Networks and the PyTorch community for providing robust deep learning tools and frameworks. And to [This User](https://github.com/cezannec/capsule_net_pytorch/blob/master/Capsule_Network.ipynb) for their great overview of routing capsule networks.
