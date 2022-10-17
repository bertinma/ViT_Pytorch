import torch
from model.transformer import ViT
import argparse
from pathlib import Path 
from torchsummary import summary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-patches', type=int, default=7)
    parser.add_argument('--hidden-dim', type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", type=str, default="weights/vit.pt")
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument('--n-blocks', type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx'])
    return parser.parse_args()

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    opt = get_args()
    # Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
    # Load model
    model = ViT(
        input_shape=(1, 28, 28),
        n_patches=opt.n_patches,
        hidden_dim=opt.hidden_dim,
        n_heads=opt.n_heads,
        out_dim=opt.n_classes,
        n_blocks=opt.n_blocks
    )
    # Evaluate the model to switch some operations from training mode to inference.
    model.eval()

    summary(model, (1, 28, 28))

    # Create dummy input for the model. It will be used to run the model inside export function.
    dummy_input = torch.randn(1, 1, 28, 28)


    model_name = Path(opt.model_path).stem
    # Call the export function
    if opt.format == 'onnx':
        torch.onnx.export(model, (dummy_input, ), f'weights/onnx/{model_name}.onnx')