import argparse
import matplotlib.pyplot as plt 

import torch 
from torch.nn import CrossEntropyLoss
from torchsummary import summary

from utils import load_dataset
from model.transformer import ViT 

torch.autograd.set_detect_anomaly(True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-patches', type=int, default=7)
    parser.add_argument('--hidden-dim', type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", type=str, default="weights/vit.pt")
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument('--display', action='store_true')
    parser.add_argument("--n-samples", type=int, default=10)
    return parser.parse_args()

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def diplay_result(x, y, y_pred):
    plt.imshow(torch.permute(x[0], (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.title(f"Actual: {y.item()}, Predicted: {y_pred.item()}")
    plt.show()

def eval(opt, model, test_loader):
    criterion = CrossEntropyLoss()
    xs = []
    ys = []
    y_preds = []
    test_correct, test_loss, test_total = 0, 0, 0
    cnt = 0
    with torch.no_grad():
        for batch in test_loader:
            cnt += 1 
            x, y = batch
            x = x.to(opt.device)
            y = y.to(opt.device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()

            test_total += 1 
            test_correct += int(torch.argmax(y_hat.data, dim=1) == y)

            xs.append(x)
            ys.append(y)
            y_preds.append(torch.argmax(y_hat.data, dim=1))
            if cnt == opt.n_samples:
                break

    print(f'Test Loss: {test_loss}, Accuracy: {test_correct/test_total*100:.2f}%')
    return xs, ys, y_preds


if __name__ == '__main__':
    # Get arguments
    opt = get_args()

    # Load MNIST dataset into DataLoader
    test_load = load_dataset(opt.batch_size, eval = True) 

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
    xs, ys, y_preds = eval(opt, model, test_load)

    # Display results
    if opt.display:
        for x, y, y_pred in zip(xs, ys, y_preds):
            diplay_result(x, y, y_pred)
        # plt.show()
