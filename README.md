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
To train the model
### Localy 
Install following dependencies : 
- torch==1.12.1
- torchsummary==1.5.1
- torchvision==0.13.1
- numpy==1.21.5
- matplotlib==3.5.1
- tqdm==4.64.1


### Using Docker 
```bash
docker build -t vit_mnist:1.0.0 .
docker run -it vit_mnist:1.0.0 -v ./weights:/app/weights -v ./datasets:/app/datasets bash
```

### Using Docker Compose 
```bash
docker-compose up -d
docker-compose exec vit_mnist sh

```

- Run the following command in bash shell opened by docker : 
```bash
python train.py --hidden-dim 8 --n-patches 7 --epochs 5 --batch-size 16  --n-heads 1 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 --n-classes 10 --device cpu
```


## Results
The model is trained on 20 epochs with a batch size of 16.
| Patches |Hidden Dims | Heads | Blocks | Test Accuracy | Parameters |
|---------|------------|-------|--------|---------------|------------|
| 7       | 8          | 1     | 1      |    61.20      | 1 242      |
| 7       | 8          | 1     | 2      |    77.04      | 2 258      |
| 7       | 8          | 1     | 4      |    68.80      | 4 290      |
| 7       | 8          | 2     | 1      |    11.35      | 1 050      |
| 7       | 8          | 2     | 2      |    35.26      | 1 874      |
| 7       | 8          | 2     | 4      |    64.71      | 3 522      |
| 14      | 8          | 1     | 1      |    11.35      | 1 146      |
| 7       | 16         | 1     | 2      |    89.57      | 8 090      |
| 7       | 16         | 1     | 4      |    91.92      | 15 738     |
| 7       | 16         | 2     | 4      |    91.92      | 12 666     |
| 7       | 32         | 1     | 4      |    91.36      | 60 138     |


We saw that more hidden dims and more blocks lead to better results.
I chose the model with better accuracy and less parameters.

Trained best model on 100 epochs with a batch size of 16.

| Patches |Hidden Dims | Heads | Blocks | Test Accuracy | Parameters |
|---------|------------|-------|--------|---------------|------------|
| 7       | 16         | 2     | 4      |  92.87        | 12 666     |

## Inference using ONNX format 
To convert the model to ONNX format, run the following command : 
```bash
python export_onnx.py --hidden-dim 16 --n-patches 7 --n-blocks 4 --n-heads 2 --n-classes 10 --model-path ./weights/vit_7p_16d_2h_4b_20e.pt
```

