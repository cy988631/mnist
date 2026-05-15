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
        output = model(img)/2
        output = torch.softmax(output,dim =1).squeeze().numpy()
        predicted = {str(i): float(output[i]) for i in range(len(output))}
    return predicted

gr.Interface(
    fn = predict,
    inputs = gr.Sketchpad(),
    outputs = gr.Label(num_top_classes = 10),
    live =True
).launch(share = True) 