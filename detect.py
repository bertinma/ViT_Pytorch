import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import torch 
from torchsummary import summary
import cv2 
from icecream import ic 

from utils import load_dataset
from model.transformer import ViT 

torch.autograd.set_detect_anomaly(True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-patches', type=int, default=7)
    parser.add_argument('--hidden-dim', type=int, default=8)
    parser.add_argument("--model-path", type=str, default="weights/vit.pt")
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=1)
    return parser.parse_args()

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def visualize(y, y_pred, **images):
    """PLot images in one row."""
    n = ic(len(images))
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

def detect(opt, model, test_loader):
    with torch.no_grad():
        cnt = 0 
        while cnt < opt.n_samples:
            batch = random.choice(test_loader.dataset)
            x, y = batch
            x = torch.unsqueeze(x, 0).to(opt.device)
            y = torch.Tensor([y]).to(opt.device)
            y_hat = model(x)
            y_pred = torch.argmax(y_hat.data, dim=1)
            attentions = compute_attentions(model.att_mat, x[0].permute(1, 2, 0).cpu().numpy())
            visualize(y, y_pred, input = x, attention_1 = attentions[0], attention_2 = attentions[1])
            cnt += 1
            print(model.att_mat.shape)

        
def compute_attentions(attention_weights, im):
    attentions = []
    for att_mat in attention_weights:
        # ic(att_mat.shape, att_mat.size(0))
        residual_att = torch.eye(ic(att_mat.size(0)))
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

if __name__ == '__main__':
    # Get arguments
    opt = get_args()

    # Load MNIST dataset into DataLoader
    test_load = load_dataset(1, eval = True) 

    # Load model
    model = ViT(
        input_shape=(1, 28, 28),
        n_patches=opt.n_patches,
        hidden_dim=opt.hidden_dim,
        n_heads=opt.n_heads,
        out_dim=opt.n_classes,
        n_blocks=opt.n_blocks
    )

    model = load_model(model, opt.model_path)

    summary(model, (1, 28, 28))

    # Eval model
    detect(opt, model, test_load)

