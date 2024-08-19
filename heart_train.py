import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from heart_dataset import heart_dataset
from heart_model import VAE, loss_function, save_model, latent_dim


"""

This script launches the training of the VAE model.

"""

# Hyperparameters
latent_dim = latent_dim   #   = 32
num_epochs = 50
batch_size = 32
learning_rate = 6e-5
l2_reg = 0.01
beta = 1.5

# Dataloader
dataloader = DataLoader(heart_dataset, batch_size=batch_size, shuffle=True)

# Model
vae_model = VAE(latent_dim=latent_dim)
optimizer = optim.AdamW(vae_model.parameters(), lr=learning_rate, weight_decay=l2_reg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae_model = vae_model.to(device)

# Training loop
vae_model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    mu_sum = 0.0
    logvar_sum = 0.0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch in progress_bar:
        batch = batch.to(device)

        # Pass the input data through the model
        recon_batch, mu, logvar = vae_model(batch)
        # mu_sum += np.sum(mu.detach().numpy())
        # logvar_sum += np.sum(logvar.detach().numpy())
        # Compute the VAE loss
        loss = loss_function(recon_batch, batch, mu, logvar, beta)
        # loss = F.binary_cross_entropy(recon_batch, batch, reduction='sum') / batch_size

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Aggregate the training loss for display at the end of the epoch
        loss_item = loss.item()
        train_loss += loss_item
        progress_bar.set_postfix({'Loss': loss_item})

    if epoch%99 == 0:
        path = 'models/vae_model_' + str(epoch) + '.pth'
        save_model(vae_model, path)

    # Display epoch information
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(dataloader):.4f}')

save_model(vae_model, 'models/vae_model.pth')