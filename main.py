import gradio as gr
from predict import *


def classify(image):
    return predict(image)


iface = gr.Interface(fn=classify,
                     inputs=gr.Image(shape=(180, 180)),
                     outputs="text")

iface.launch(share=True)
