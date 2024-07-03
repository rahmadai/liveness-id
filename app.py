import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
from facedb import FaceDB
import torchvision


app = FastAPI()

# Load your model and set up transformations
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('model/model.pth', map_location=device))
model = model.to(device)
model.eval()

# Create a FaceDB instance
db = FaceDB(path="facedata")

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

# Define request body model for image path
class ImagePath(BaseModel):
    path: str

# Prediction endpoint
@app.post("/predict/")
async def predict(image_path: ImagePath):
    try:
        # Make prediction
        prediction = predict_image(image_path.path, model, inference_transform, device)

        result_prediction = None
        if prediction == 0:
            result_prediction = "Spoof"
        else:
            result_prediction = "Live"
        
        # Search image in database
        result_search = search_image(image_path.path)
        
        return {
            "image_path": image_path.path,
            "predicted_class": result_prediction,
            "result_search": result_search
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
