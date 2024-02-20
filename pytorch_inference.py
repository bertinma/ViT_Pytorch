import torch 
import os
import argparse
import gradio as gr 
import numpy as np 
import cv2
import time 


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX model to OpenVINO IR')
    parser.add_argument('--model',  type= str, required = True ,help='Path to ONNX model')
    parser.add_argument('--device', type= str, default = 'CPU', help='device to use, cpu or tpu')
    parser.add_argument('--source', type= str, choices = ['images', 'camera'], default = 'images', help='source of images')
    return parser.parse_args()

def vit_classifier_image(image_path):
    """
    Process inference for rondelles
    Args:
        - image
    Returns:
        - segmentation mask"""
    # Preprocess image
    # image = PIL.Image.resize(image, (28, 28))
    t0 = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.astype(np.float32) / 255.
    input_image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    t1 = time.time()
    result = vit(input_image)
    t2 = time.time()

    print('Preprocess time: ', t1 - t0)
    print('Inference time: ', t2 - t1)
    print('Total time: ', t2 - t0)

    # Prepare data for visualization
    prediction = np.argmax(result.detach().numpy(), axis=1)[0]
    return prediction

if __name__ == '__main__':
    args = parse_args()

    # print(input_layer, output_layer)

    vit = torch.load(args.model)
    print(vit)
    with torch.no_grad():

        if args.source == 'images':
            title = "ViT MNIST Classifier" 
            description = "Classify MNIST digits using ViT"
            iface = gr.Interface(
                vit_classifier_image,
                [
                    gr.components.Image(
                        height=None,
                        width=None,
                        image_mode="L",  
                        # invert_colors=False,
                        sources="upload",
                        # tool="editor",
                        type="filepath",
                        label='MNIST Image'),
                ],
                [
                    gr.components.Textbox(type="text", label='Prediction'),
                ],
                title=title,
                description=description,
                )

            iface.launch(server_name="0.0.0.0", server_port=int(os.getenv('PORT', "8150")))
        elif args.source == 'camera':
            stream = cv2.VideoCapture(0)
            while True:
                start_time = time.time() # We would like to measure the FPS.
                ret, frame = stream.read()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if ret:
                    frame = cv2.resize(frame, (28, 28))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = frame.astype(np.float32) / 255.
                    input_image = np.expand_dims(np.expand_dims(frame, axis=0), axis=0)
                    result = vit_classifier_image(frame)
                    prediction = np.argmax(result.detach().numpy(), axis=1)[0]
                    end_time = time.time()   
                    fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
                    cv2.putText(frame, str(prediction), (2, 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(frame, str(fps), (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    cv2.imshow('frame', frame)

