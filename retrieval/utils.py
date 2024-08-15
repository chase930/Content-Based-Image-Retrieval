from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def extract_feature(image_path, feature_extractor, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  
    with torch.no_grad():
        feature = feature_extractor(image_tensor)
    feature_np = feature.squeeze().numpy()
    feature_np = feature_np / np.linalg.norm(feature_np)
    return feature_np
