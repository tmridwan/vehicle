# --- Fix Windows pickle paths BEFORE anything else ---
import pathlib
pathlib.WindowsPath = pathlib.PosixPath

# --- Imports ---
import gradio as gr
from fastai.vision.all import load_learner, PILImage
from huggingface_hub import hf_hub_download

# --- Class labels ---
VEHICLE_LABELS = [
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

# --- Lazy-loaded model ---
_learner = None

def load_model():
    global _learner
    if _learner is None:
        model_path = hf_hub_download(
            repo_id="tmridwan03/vehicle-classification-model",
            filename="model.pkl"
        )

        _learner = load_learner(model_path)

        # Move to CPU safely
        if hasattr(_learner, "dls"):
            _learner.dls.cpu()

    return _learner

# --- Prediction function ---
def recognize_image(image):
    learner = load_model()
    img = PILImage.create(image)

    _, _, probs = learner.predict(img)

    top_probs, top_idxs = probs.topk(5)
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    return {
        VEHICLE_LABELS[int(i)]: float(p)
        for p, i in zip(top_probs, top_idxs)
    }

# --- Gradio UI ---
demo = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top Predictions"),
    title="🚗 Vehicle Classification",
    description="Upload an image to classify the vehicle type."
)

# --- Launch (HF Spaces compatible) ---
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
