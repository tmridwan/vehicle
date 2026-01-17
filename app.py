from fastai.vision.all import load_learner

learn = load_learner("vehicle_model.pkl", cpu=True)

print("TYPE:", type(learn))
print("Has predict:", hasattr(learn, "predict"))
print("Has dls:", hasattr(learn, "dls"))
print("Dir sample:", [x for x in dir(learn) if x in ["dls","tls","model","predict","vocab"]])
