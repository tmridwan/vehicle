# --- Fix Windows pickle paths BEFORE anything else ---
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

import io
import gradio as gr
from PIL import Image, ImageOps
from fastai.vision.all import load_learner, PILImage
import torch

_learner = None


def load_model():
    global _learner
    if _learner is None:
        _learner = torch.load("vehicle_model.pkl", map_location="cpu")
    return _learner

import io
from PIL import Image, ImageOps, UnidentifiedImageError

def decode_image(file_path: str) -> Image.Image:
    with open(file_path, "rb") as f:
        raw = f.read()

    # Debug info (shows in Space logs)
    print("Uploaded file:", file_path)
    print("File size:", len(raw))
    print("First bytes:", raw[:16])

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except UnidentifiedImageError:
        raise gr.Error(
            "❌ Unsupported image format. "
            "Please upload a standard JPG or PNG image "
            "(not HEIC / AVIF / WEBP)."
        )


def recognize_image(file):
    learner = load_model()

    # gr.File returns a path or object
    file_path = file if isinstance(file, str) else file.name

    pil_img = decode_image(file_path)

    fa_img = PILImage.create(pil_img)

    _, _, probs = learner.predict(fa_img)

    vocab = list(learner.dls.vocab) if hasattr(learner, "dls") else None

    top_probs, top_idxs = probs.topk(5)
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    if vocab:
        return {vocab[int(i)]: float(p) for p, i in zip(top_probs, top_idxs)}

    return {f"class_{int(i)}": float(p) for p, i in zip(top_probs, top_idxs)}


demo = gr.Interface(
    fn=recognize_image,
    inputs=gr.File(file_types=["image"], label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top Predictions"),
    title="🚗 Vehicle Classification",
    description="Upload an image to classify the vehicle type."
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
