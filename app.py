import pathlib
# ✅ Fix Windows-exported fastai pickle on Linux
pathlib.WindowsPath = pathlib.PosixPath

import gradio as gr
from fastai.vision.all import load_learner, PILImage

MODEL_PATH = "vehicle_model.pkl"

learn = None
vehicle_labels = None

def get_model():
    global learn, vehicle_labels
    if learn is None:
        learn = load_learner(MODEL_PATH)   # don't use cpu=True here
        # move dataloaders to cpu only if they exist
        if hasattr(learn, "dls"):
            learn.dls.cpu()
            vehicle_labels = list(learn.dls.vocab)
        else:
            # fallback if vocab stored elsewhere
            vehicle_labels = getattr(learn, "vocab", None)
    return learn

def recognize_image(image):
    model = get_model()
    img = PILImage.create(image)

    pred, idx, probs = model.predict(img)

    # return top 5 only (clean + fast)
    top_probs, top_idxs = probs.topk(5)
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    return {vehicle_labels[int(i)]: float(p) for p, i in zip(top_probs, top_idxs)}

with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Vehicle Classifier")

    img = gr.Image(type="pil", label="Upload image")
    out = gr.Label(num_top_classes=5, label="Prediction")

    btn = gr.Button("Submit")
    btn.click(fn=recognize_image, inputs=img, outputs=out)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
