from fastai.vision.all import load_learner
import gradio as gr
from PIL import Image

# Load model (FastAI-safe)
model = load_learner("vehicle_model.pkl", cpu=True)

vehicle_labels = model.dls.vocab

def recognize_image(image: Image.Image):
    pred, idx, probs = model.predict(image)
    return {
        vehicle_labels[i]: float(probs[i])
        for i in range(len(vehicle_labels))
    }

with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Vehicle Classifier")

    img = gr.Image(type="pil", label="Upload image")
    out = gr.Label(num_top_classes=5)

    btn = gr.Button("Submit")
    btn.click(fn=recognize_image, inputs=img, outputs=out)

demo.launch()
