import gradio as gr
from PIL import Image

def predicted(img):
    composite = img["composite"]
    print(type(composite))
    print(composite.shape)
    # print(img.shape)
    # print(img.dtype)
    return"test"

gr.Interface(
    fn = predict,
    inputs = gr.Sketchpad(),
    outputs = gr.Text()
).launch(share = True) 