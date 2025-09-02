import torch
import torch.nn as nn
import cv2
import os
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"   
CLASS_NAMES = ["Fake", "Real"]

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(image=img)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][pred].item()

    print(f"🖼️ {os.path.basename(image_path)} -> {CLASS_NAMES[pred]} ({prob*100:.2f}% confidence)")

if __name__ == "__main__":
  
    folder_path = r"C:\Users\sajja\Desktop\Deepfake Mini project\test_images"   # replace with your folder path
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                predict_image(os.path.join(folder_path, file))
