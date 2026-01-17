import gradio as gr
from fastai.vision.all import *
from huggingface_hub import hf_hub_download
import pathlib

# Fix Windows -> Linux paths for pickled fastai learner
pathlib.WindowsPath = pathlib.PosixPath

vehicle_labels = [
    "Pickup truck", "SUV", "Van / Minivan", "Station wagon",
    "Convertible", "Sports car", "Hatchback", "Coupe", "Sedan",
    "Limousine", "Taxi", "Police car", "Ambulance", "Fire truck",
    "Light truck", "Heavy truck", "Semi truck", "Tow truck",
    "Garbage truck", "Cement mixer truck", "Dump truck",
    "Refrigerated truck", "Flatbed truck", "Tanker truck",
    "Bus", "Mini-bus", "School bus", "Coach bus",
    "Tram", "Train", "Subway train",
    "Bicycle", "Motorcycle", "Scooter", "Moped", "Dirt bike",
    "Three-wheeler", "Tricycle",
    "Tractor", "Combine harvester", "Bulldozer", "Excavator",
    "Backhoe loader", "Skid steer", "Forklift",
    "Road roller", "Crane",
    "Airplane", "Helicopter", "Glider", "Hot air balloon",
    "Drone", "Jet", "Cargo aircraft", "Seaplane",
    "Boat", "Ship", "Sailboat", "Yacht", "Speedboat",
    "Fishing boat", "Cargo ship", "Cruise ship",
    "Submarine", "Kayak", "Jet ski",
    "Carriage", "Rickshaw", "Handcart",
    "Wheelchair", "Skateboard", "Roller skates"
]

learn = None

def get_model():
    global learn
    if learn is None:
        # If your model is in the Space repo, keep this:
        # model_path = "vehicle_model.pkl"

        # If your model is in HF model repo, use this instead:
        model_path = hf_hub_download(
            repo_id="tmridwan03/vehicle-classification-model",
            filename="model.pkl"
        )

        learn = load_learner(model_path)  # don't force cpu=True here
        if hasattr(learn, "dls"):         # avoid dls crash if missing
            learn.dls.cpu()
    return learn

def recognize_image(image):
    model = get_model()
    img = PILImage.create(image)
    pred, idx, probs = model.predict(img)

    top_probs, top_idxs = probs.topk(5)
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    return {vehicle_labels[int(i)]: float(p) for p, i in zip(top_probs, top_idxs)}

demo = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Label(num_top_classes=5),
    examples=["b.jpg", "c.jpg", "h.jpg", "e.png"],  # only if these files exist in repo
    title="Vehicle Classifier"
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
