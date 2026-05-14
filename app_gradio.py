import gradio as gr
from PIL import Image
from model_cnn import CNN
import numpy as np
import torch
import cv2

model = CNN().to('cpu')
model.load_state_dict(torch.load("mnist_CNN_model.pth",map_location = 'cpu'))
model.eval()

def predict(input_data):
    img = input_data["composite"]
    img = np.dot(img[..., :3],[0.299,0.587,0.114])
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    img = 255 - img
    Image.fromarray(img.astype(np.uint8)).save('debug.png')
    img = img/255.0
    img = (img-0.1307)/0.3081
    img = torch.tensor(img).float()
    img = img.view(1,1,28,28)
    
    with torch.no_grad():
        output = model(img)
        predicted = output.argmax(dim = 1)
    return str(predicted.item())

gr.Interface(
    fn = predict,
    inputs = gr.Sketchpad(),
    outputs = gr.Text()
).launch(share = True) 