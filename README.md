# Vision Transformer 

This repository implements a simple clissifer based on Vision Transformer 

This repo uses PyTorch framework. 

## Model
<!-- <img src="images/vit.png"/> -->

Classic Transformer architecture as follow : 
* Patch embedding (with convolution) + Positionnal embedding (learnable) 
* Normalization 1 
* Multihead Attention Layer 
* Residual Connection 
* Normalization 2 
* Multilayer Perceptron with : 
    * Dense Layer 
    * GELU activation function (https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c)
    * Dropout 
    * Dense Layer 
    * Dropout 
* Head with : 
    * Normalization
    * Linear Layer 

## Data 
We use MNIST Fashion dataset 

