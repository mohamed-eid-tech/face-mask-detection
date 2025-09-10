import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('face_mask_detection.h5', compile=False)

def predict_mask(image):
    img = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    pred = model.predict(img_array)[0][0]
    percentage = pred * 100  

    if pred >= 0.5:
        return f"With Mask 😷 ({percentage:.2f}%)"
    else:
        return f"Without Mask ❌ ({100 - percentage:.2f}%)"

iface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Mask Detection",
    description="Upload an image to check if the person is wearing a mask."
)

iface.launch()
