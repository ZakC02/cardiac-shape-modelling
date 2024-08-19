import torch
from torch.utils.data import DataLoader
from heart_display import plot_reconstructions
from heart_model import VAE
from heart_dataset import HeartDataset
from heart_variables import device, output_path
from heart_display import load_images
import numpy as np

# Dice coefficient
def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=1) + target.sum(dim=2).sum(dim=1) + smooth)
    
    return dice.mean().item()

def compute_mean_dice(vae_model, dataloader):
    dice_scores = []

    vae_model.eval()  
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            recon_batch, _, _ = vae_model(batch, testing=True)
            
            preds = torch.sigmoid(recon_batch) > 0.5
            dice = dice_coefficient(preds, batch)
            dice_scores.append(dice)

    # Average dice
    average_dice_score = np.mean(dice_scores)
    return average_dice_score

