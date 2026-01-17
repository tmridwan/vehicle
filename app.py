from fastai.vision.all import *
import gradio as gr
import torch


vehicle_labels = [
    # Cars & Light Vehicles
    "Pickup truck", "SUV", "Van / Minivan", "Station wagon",
    "Convertible", "Sports car", "Hatchback", "Coupe", "Sedan", 
    "Limousine", "Taxi", "Police car", "Ambulance", "Fire truck",

    # Trucks & Utility
    "Light truck", "Heavy truck", "Semi truck", "Tow truck", 
    "Garbage truck", "Cement mixer truck", "Dump truck",
    "Refrigerated truck", "Flatbed truck", "Tanker truck",

    # Mass Transport
    "Bus", "Mini-bus", "School bus", "Coach bus", 
    "Tram", "Train", "Subway train",

    # Two- & Three-Wheelers
    "Bicycle", "Motorcycle", "Scooter", "Moped", "Dirt bike",
    "Three-wheeler", "Tricycle",

    # Agriculture & Construction
    "Tractor", "Combine harvester", "Bulldozer", "Excavator",
    "Backhoe loader", "Skid steer", "Forklift", 
    "Road roller", "Crane",

    # Aircraft
    "Airplane", "Helicopter", "Glider", "Hot air balloon",
    "Drone", "Jet", "Cargo aircraft", "Seaplane",

    # Watercraft
    "Boat", "Ship", "Sailboat", "Yacht", "Speedboat",
    "Fishing boat", "Cargo ship", "Cruise ship",
    "Submarine", "Kayak", "Jet ski",

    # Non-Motorized / Animal-Drawn
    "Carriage", "Rickshaw", "Handcart", 
    "Wheelchair", "Skateboard", "Roller skates"
]

# Fix for cross-platform compatibility (Windows -> Linux)
import pathlib
import platform

# Monkey-patch WindowsPath for Linux compatibility
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Load model with torch, allowing fastai classes
model = torch.load('vehicle_model.pkl', map_location='cpu', weights_only=False)

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(vehicle_labels, map(float, probs)))

# Create Gradio interface with modern API
demo = gr.Interface(
    fn=recognize_image, 
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=[
        'b.jpg',
        'c.jpg',
        'h.jpg',
        'e.png'
    ],
    title="🚗 Vehicle Classification",
    description="Upload an image of a vehicle to get AI-powered predictions for 60+ vehicle types!"
)

if __name__ == "__main__":
    demo.launch()