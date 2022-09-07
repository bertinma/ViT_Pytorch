from d2l import torch as d2l
from model import transformer
import argparse
import torch 

def train(opt):
    pass

def save_model(model, path="weights/vit.pt"):
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=96, help="Size of image")
    parser.add_argument('--patch-size', type=int, default=16, help="Size of patches")
    parser.add_argument('--num-hiddens', type=int, default=512)
    parser.add_argument('--mlp-num-hiddens', type=int, default=2048)
    parser.add_argument('--num-heads', type=int, default=8, help="Number of head attentions")
    parser.add_argument('--num-blocks', type=int, default=2, help="Number of blocks")
    parser.add_argument('--emb-dropout', type=float, default=.1, help="Embedded dropout")
    parser.add_argument('--block-dropout', type=float, default=.1, help="Block dropout")
    parser.add_argument('--lr', type=float, default=.1, help="Learning rate")
    opt = parser.parse_args()
    
    model = train(opt)
    save_model(model)