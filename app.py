import gradio as gr
from fastai.vision.all import load_learner
import pathlib, platform

# Fix Windows → Linux path issue
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

# Load model safely (CPU only)
model = load_learner("vehicle_model.pkl", cpu=True)

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(model.dls.vocab, map(float, probs)))

gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="🚗 Vehicle Recognizer",
    description="Upload an image to identify the vehicle"
).launch()
