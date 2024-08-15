import torch
import torchvision.models as models

def load_feature_extractor():
    model = models.resnet50(pretrained=True)
    model.eval()
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    return feature_extractor
