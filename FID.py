import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import InterpolationMode

# Function to extract features using InceptionV3
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = inception_v3(pretrained=True)
        self.features = nn.Sequential(*list(inception.children())[:-1])  # Remove the classification layer

    def forward(self, x):
        x = self.features(x)
        return x.squeeze()
    
def preprocess_1(images):
    # Step 1: Compute the argmax along the channel dimension
    argmax_tensor = torch.argmax(images, dim=1)
    
    # Step 2: Subtract one from the argmax values
    subtracted_tensor = argmax_tensor - 1
    
    # Step 3: Apply ReLU to ensure values are within {0, 1, 2}
    relu_tensor = F.relu(subtracted_tensor)
    
    # Step 4: One-hot encode the resulting tensor into 3 channels
    one_hot_tensor = F.one_hot(relu_tensor, num_classes=3).permute(0,3,1,2).float()
    
    return one_hot_tensor

def preprocess_2(images):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299), interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = torch.stack([preprocess(img) for img in images])
    return images


def get_features(dataloader, model, device):
    model.eval()
    features = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch = preprocess_1(batch).to(device)
            batch = preprocess_2(batch).to(device)
            output = model(batch)
            features.append(output.cpu().numpy())

    return np.concatenate(features, axis=0)

# Calculate statistics and FID score
def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2):
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2*covmean)