from heart_display import load_images, display_images
from heart_variables import output_path
from heart_dataset import HeartDataset
from tqdm import tqdm
import heart_model as HM
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_new_images(model, latent_dim, number_of_images):
    latent_vectors = torch.rand(number_of_images, latent_dim).to(device)
    new_images = model.decoder(latent_vectors)
    return new_images