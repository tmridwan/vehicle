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

# Load model with torch, allowing fastai classes
try:
    model = torch.load('vehicle_model.pkl', map_location='cpu', weights_only=False)
except Exception as e:
    # Fallback: try with safe_globals
    from pathlib import PosixPath, WindowsPath
    torch.serialization.add_safe_globals([WindowsPath, PosixPath])
    model = torch.load('vehicle_model.pkl', map_location='cpu', weights_only=False)

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(vehicle_labels, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'b.jpg',
    'c.jpg',
    'h.jpg',
    'e.png'
    ]


iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)