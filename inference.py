import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from PIL import Image
from facedb import FaceDB

# Create a FaceDB instance
db = FaceDB(
    path="facedata")


# Define the transformation for inference
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Modify this line to handle CUDA availability
model.load_state_dict(torch.load('model/model.pth', map_location=device))

model = model.to(device)
model.eval()

# Function to make predictions
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    return preds.item()

def search_image(path):
    # Recognize a face
    result = db.recognize(img=path, include="name")

    return result["name"]



# Example usage
image_path = "/Users/rahmad/work/others/facerecs/imglist/bob_7.png"
prediction = predict_image(image_path, model, inference_transform, device)
print(f'Predicted class: {prediction}')
result_search = search_image(image_path)
print(result_search)
