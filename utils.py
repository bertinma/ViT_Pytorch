from torch.utils.data import DataLoader

from torchvision.datasets.mnist import MNIST 
from torchvision.transforms import ToTensor
import torch
import cv2 
import matplotlib.pyplot as plt
import numpy as np 

def load_dataset(batch_size, eval = False):
    transform = ToTensor()
    if not eval:
        train_set = MNIST(root='./datasets/', train=True, download=True, transform=transform)
        train_load = DataLoader(train_set, shuffle=True, batch_size=batch_size)   
        shuffle = False 
    else:
        batch_size=1 
        shuffle = True

    test_set = MNIST(root='./datasets/', train=False, download=True, transform=transform)
    test_load = DataLoader(test_set, shuffle=shuffle, batch_size=batch_size)

    if not eval:
        return train_load, test_load
    else:
        return test_load


def compute_attentions(attention_weights, im):
    attentions = []
    for att_mat in attention_weights:
        # ic(att_mat.shape, att_mat.size(0))
        residual_att = torch.eye(att_mat.size(0))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        v = joint_attentions
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0,1:].reshape(grid_size, grid_size).detach().numpy()
        attentions.append(cv2.resize(mask / mask.max(), (28, 28))[..., np.newaxis])
    return attentions

def visualize(y, y_pred, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    if y.item() == y_pred.item():
        color = 'green'
    else:
        color = "red"
    for i, (name, image) in enumerate(images.items()):
        if name == 'input':
            image = image[0, 0, :, :]
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title(), color = color)
        plt.imshow(image)
    plt.show()