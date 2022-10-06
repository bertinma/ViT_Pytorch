import argparse
from tqdm import tqdm, trange
import torch 
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchsummary import summary

from utils import load_dataset
from model.transformer import ViT 

torch.autograd.set_detect_anomaly(True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-patches', type=int, default=7)
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="weights/vit.pt")
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def train(opt, model, train_loader, test_loader):
    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = CrossEntropyLoss()

    for epoch in trange(opt.epochs, desc = "Training"):
        train_loss = 0.0
        correct, total = 0, 0 
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x = x.to(opt.device)
            y = y.to(opt.device)

            y_hat = model(x)
            loss = criterion(y_hat, y) / len(x)
            train_loss += loss.item()

            total += len(x)
            correct += torch.sum((torch.argmax(y_hat.data, dim=1) == y)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        print(f'\nEpoch: {epoch+1}/{opt.epochs}, Train Loss: {train_loss}, Train Accuracy: {correct/total*100:.2f}%')
        test_correct, test_total = 0, 0 
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing', leave=False):
                x, y = batch
                x = x.to(opt.device)
                y = y.to(opt.device)

                y_hat = model(x)
                loss = criterion(y_hat, y) / len(x)
                test_loss += loss.item()

                test_total += len(x)
                test_correct += torch.sum((torch.argmax(y_hat.data, dim=1) == y)).item()

        print(f'\nTest Loss: {test_loss}, Accuracy: {test_correct/test_total*100:.2f}%')
        
    model_name = f"weights/vit_{opt.n_patches}_{opt.hidden_dim}_{opt.n_heads}_{opt.epochs}.pt"
    save_model(model, model_name)


def save_model(model, path="weights/vit.pt"):
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    # Get arguments
    opt = get_args()

    # Load MNIST dataset into DataLoader
    train_load, test_load = load_dataset(opt.batch_size) 

    # Load model
    model = ViT(
        input_shape=(1, 28, 28),
        n_patches=opt.n_patches,
        hidden_dim=opt.hidden_dim,
        n_heads=opt.n_heads,
        out_dim=opt.n_classes
    )

    print(summary(model, (1, 28, 28)))

    # Train model
    train(opt, model, train_load, test_load)