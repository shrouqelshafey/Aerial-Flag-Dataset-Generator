import numpy, cv2, PIL, yaml, matplotlib

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", PIL.__version__)
print("PyYAML:", yaml.__version__)
print("Matplotlib:", matplotlib.__version__)

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f) 
print("Config loaded. Keys found:", list(cfg.keys())) 