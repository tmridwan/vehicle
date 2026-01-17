# --- Fix Windows pickle paths BEFORE anything else ---
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

import gradio as gr
from fastai.vision.all import load_learner, PILImage

_learner = None

def load_model():
    global _learner
    if _learner is None:
        _learner = load_learner("vehicle_model.pkl")
        if hasattr(_learner, "dls"):
            _learner.dls.cpu()
    return _learner

def recognize_image(image):
    learner = load_model()
    img = PILImage.create(image)

    _, _, probs = learner.predict(img)

    vocab = list(learner.dls.vocab) if hasattr(learner, "dls") else None

    top_probs, top_idxs = probs.topk(5)
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    if vocab:
        return {vocab[int(i)]: float(p) for p, i in zip(top_probs, top_idxs)}
    return {f"class_{int(i)}": float(p) for p, i in zip(top_probs, top_idxs)}

demo = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top Predictions"),
    title="🚗 Vehicle Classification",
    description="Upload an image to classify the vehicle type."
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
