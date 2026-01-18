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

from fastai.vision.all import load_learner
import pathlib, platform

# Fix Windows → Linux paths
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

model = load_learner("vehicle_model.pkl", cpu=True)

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(vehicle_labels, map(float, probs)))

image = gr.Image(shape=(192,192))
label = gr.Label(num_top_classes=5)
examples = ['b.jpg', 'c.jpg', 'h.jpg', 'e.png']



iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)