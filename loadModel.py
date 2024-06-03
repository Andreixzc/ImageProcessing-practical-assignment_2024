import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset  # Added import statement
from PIL import Image
import pandas as pd

# Define your own Dataset to load images and labels from the CSV
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(self.data['Label'].unique())}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path = self.data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 1]
        label = self.label_to_idx[label]  # Convert the label to a numeric index

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()  # Convert the label to tensor

        return image, label

# Path to the CSV file
csv_file = 'image_paths_labels.csv'

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset (needed to define num_classes)
dataset = CustomDataset(csv_file, transform=transform)

# Load the trained model
def load_model(model_path, num_classes):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define the device for prediction (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Path to the saved model
model_path = 'model.pth'

# Number of classes in your dataset
num_classes = len(set(dataset.data.iloc[:, 1]))  # Calculate the number of classes

# Load the trained model
model = load_model(model_path, num_classes)
model.to(device)

# Function to predict the label of a single image
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# List of image paths to predict
image_paths = ['28-05-2024/ASC-H/10.png', '28-05-2024/ASC-H/11.png', '28-05-2024/ASC-H/12.png']  # Add your image paths here

# Load label to index mapping from your dataset
label_to_idx = {label: idx for idx, label in enumerate(dataset.data['Label'].unique())}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}  # Invert the dictionary to map index to label

# Predict the label for each image
for image_path in image_paths:
    predicted_idx = predict_image(image_path, model, transform, device)
    predicted_label = idx_to_label[predicted_idx]
    print(f'Image: {image_path}, Predicted Label: {predicted_label}')
