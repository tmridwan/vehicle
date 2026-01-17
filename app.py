# --- Fix Windows pickle paths BEFORE anything else ---
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

# --- Imports ---
import io
import gradio as gr
from PIL import Image, ImageOps
from fastai.vision.all import load_learner, PILImage
from huggingface_hub import hf_hub_download

# --- Lazy-loaded model ---
_learner = None

def load_model():
    global _learner
    if _learner is None:
        model_path = hf_hub_download(
            repo_id="tmridwan03/vehicle_recognizer",
            filename="vehicle_model.pkl"
        )
        _learner = load_learner(model_path)
        if hasattr(_learner, "dls"):
            _learner.dls.cpu()
    return _learner

def decode_image(file_path: str) -> Image.Image:
    """Robust image decode: handles EXIF rotation + forces RGB."""
    with open(file_path, "rb") as f:
        raw = f.read()

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img

# --- Prediction function ---
def recognize_image(file):
    learner = load_model()

    # file can be a string path or an object (depending on gradio version)
    file_path = file if isinstance(file, str) else file.name

    pil_img = decode_image(file_path)
    fa_img = PILImage.create(pil_img)

    _, _, probs = learner.predict(fa_img)

    # Get labels from the model (real vocab)
    vocab = list(learner.dls.vocab) if hasattr(learner, "dls") else None

    top_probs, top_idxs = probs.topk(5)
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    if vocab:
        return {vocab[int(i)]: float(p) for p, i in zip(top_probs, top_idxs)}
    else:
        return {f"class_{int(i)}": float(p) for p, i in zip(top_probs, top_idxs)}

# --- Gradio UI ---
demo = gr.Interface(
    fn=recognize_image,
    inputs=gr.File(file_types=["image"], label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top Predictions"),
    title="🚗 Vehicle Classification",
    description="Upload an image to classify the vehicle type."
)

# --- Launch (HF Spaces compatible) ---
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
