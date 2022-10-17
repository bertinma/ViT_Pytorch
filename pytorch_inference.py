import torch 
import os
import argparse
import gradio as gr 
import numpy as np 
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX model to OpenVINO IR')
    parser.add_argument('--model',  type= str, required = True ,help='Path to ONNX model')
    parser.add_argument('--device', type= str, default = 'CPU', help='device to use, cpu or tpu')
    return parser.parse_args()

def vit_classifier(image_path):
    """
    Process inference for rondelles
    Args:
        - image
    Returns:
        - segmentation mask"""
    # Preprocess image
    # image = PIL.Image.resize(image, (28, 28))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.astype(np.float32) / 255.
    input_image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    result = vit(input_image)
    # Prepare data for visualization
    prediction = np.argmax(result.detach().numpy(), axis=1)[0]
    return prediction

if __name__ == '__main__':
    args = parse_args()

    # print(input_layer, output_layer)

    vit = torch.load(args.model)
    vit.eval()

    title = "ViT MNIST Classifier" 
    description = "Classify MNIST digits using ViT"
    iface = gr.Interface(
        vit_classifier,
        [
            gr.components.Image(
                shape=None,
                image_mode="L",  
                invert_colors=False,
                source="upload",
                tool="editor",
                type="filepath",
                label='MNIST Image'),
        ],
        [
            gr.components.Textbox(type="auto", label='Prediction'),
        ],
        title=title,
        description=description,
        )

    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv('PORT', "8150")))


