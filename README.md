# Vision Transformer 

This repository implements a simple clissifer based on Vision Transformer 

This repo uses PyTorch framework. 

## Model
<!-- <img src="images/vit.png"/> -->

Classic Transformer architecture as follow : 
* Linear layer for tokenization of patches (previously reshaped)
* Concatenation of classificatioon token
* Positionnal embedding (with sin and cos functions)
* Normalization 1 
* Multihead Attention Layer 
* Residual Connection 
* Normalization 2 
* Multilayer Perceptron with : 
    * Linear Layer 
    * GELU activation function (https://medium.com/@shauryagoel/gelu-gaussian-error-linear-unit-4ec59fb2e47c)
    * Dropout 
    * Dense Layer 
    * Dropout 
* Claddification Head with : 
    * Normalization
    * Linear Layer (and softmax activation function)

## Data 
We use MNIST Fashion dataset 

## Training
To train the model, run the following command : 
```bash
python train.py --hidden-dim 8 --n-patches 7 --epochs 5 --batch-size 16  --n-heads 1 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 --n-classes 10 --device cpu
```

## Results
The model is trained on 5 epochs with a batch size of 16.
| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|-------|------------|----------------|-----------|---------------|
| 1     | 0.0000     | 0.0000         | 0.0000    | 0.0000        |
| 2     | 0.0000     | 0.0000         | 0.0000    | 0.0000        |
| 3     | 0.0000     | 0.0000         | 0.0000    | 0.0000        |
| 4     | 0.0000     | 0.0000         | 0.0000    | 0.0000        |
| 5     | 0.0000     | 0.0000         | 0.0000    | 0.0000        |
