import argparse
import random 
import torch 
from torchsummary import summary

from utils import load_dataset, compute_attentions, visualize
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

