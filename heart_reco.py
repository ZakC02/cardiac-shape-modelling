import torch
from torch.utils.data import DataLoader
from heart_display import plot_reconstructions
from heart_model import VAE, latent_dim
from heart_dataset import dataloader, HeartDataset
from heart_variables import device
from heart_display import load_images

vae_model = VAE(latent_dim=latent_dim).to(device)
vae_model.load_state_dict(torch.load('models/vae_model_99.pth'), map_location=torch.device(device))

# Get a batch from the dataloader
vae_model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        # Pass the input data through the model
        recon_batch, _, _ = vae_model(batch)

        # Display the original images and their reconstructions
        plot_reconstructions(batch, recon_batch)
        break  # Display only the first batch



images_test, names_test = load_images('all_patient_images_test')

heart_dataset_test = HeartDataset(images_test)
dataloader_test = DataLoader(heart_dataset_test, batch_size=32, shuffle=True)

original_images_test = []
reconstructed_images_test = []
vae_model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for batch in dataloader_test:
        original_images_test.extend([elt.cpu() for elt in batch])
        batch = batch.to(device)
        # Pass the input data through the model
        recon_batch, _, _ = vae_model(batch, testing=True)
        reconstructed_images_test.extend([elt.cpu() for elt in recon_batch])

original_images_test = torch.stack(original_images_test)
reconstructed_images_test = torch.stack(reconstructed_images_test)

plot_reconstructions(original_images_test, reconstructed_images_test, n=20)