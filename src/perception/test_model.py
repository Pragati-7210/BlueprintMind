import torch

from model import build_model


MODEL_PATH = "weights/best.safetensors"


model = build_model(MODEL_PATH)

print("Model loaded successfully!")
print("Parameters:", sum(p.numel() for p in model.parameters()))

x = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)

