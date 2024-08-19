import os
import torch
import torchio as tio
import matplotlib.pyplot as plt
from heart_variables import base_path, output_path



def load_images(output_path):
    # Get the list of all image files
    image_files = sorted([f for f in os.listdir(output_path) if f.endswith('.nii.gz')])
    images = []
    names = []

    for image_file in image_files:
        # Load the image using torchio
        image_path = os.path.join(output_path, image_file)
        image = tio.LabelMap(image_path)

        # Extract the data for the first channel and convert to numpy array
        image_data = image.data.squeeze().numpy()
        images.append(image_data)
        names.append(image_path)

    return images, names

#output_path = 'all_patient_images'
#images, names = load_images(output_path)

def display_images(output_path, start_idx, end_idx):
    # Get the list of all image files
    image_files = sorted([f for f in os.listdir(output_path) if f.endswith('.nii.gz')])

    # Validate indices
    if start_idx < 0 or end_idx > len(image_files) or start_idx >= end_idx:
        raise ValueError("Invalid start or end index")

    # Calculate the number of images to display
    num_images_to_display = end_idx - start_idx
    num_rows = (num_images_to_display + 4) // 5  # Calculate the number of rows needed (5 images per row)

    fig, axes = plt.subplots(num_rows, 5, figsize=(20, num_rows * 4))

    for i, image_file in enumerate(image_files[start_idx:end_idx]):
        # Load the image using torchio
        image_path = os.path.join(output_path, image_file)
        image = tio.LabelMap(image_path)

        # Extract the data for the first channel (assuming single-channel image)
        image_data = image.data.squeeze()  # Remove singleton dimensions

        # Display the image
        ax = axes.flat[i]
        ax.imshow(image_data, cmap='gray')
        ax.set_title(f'Image: {start_idx + i + 1}')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes.flat)):
        axes.flat[j].axis('off')

    plt.tight_layout()
    plt.show()



def plot_reconstructions(original_images, reconstructed_images, n=10):
    """
    Plots the original and reconstructed images side by side.

    Args:
        original_images (torch.Tensor): Original images batch, shape (B, C, H, W)
        reconstructed_images (torch.Tensor): Reconstructed images batch, shape (B, C, H, W)
        n (int): Number of images to display
    """
    # Make sure n is not greater than the batch size
    n = min(n, len(original_images))

    # Convert one-hot encoded images to single-channel images
    original_images = torch.argmax(original_images.permute(0,2,3,1), dim=3)
    reconstructed_images = torch.argmax(reconstructed_images.permute(0,2,3,1), dim=3)

    # Create a figure
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # Original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].cpu().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title("Original")

        # Reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i].cpu().detach().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_title("Reconstructed")

    plt.show()

# Define the output path where images are saved
#output_path = 'all_patient_images'




# Define start and end indices
#start_idx = 1
#end_idx = 100



# Display the images
#display_images(output_path, start_idx, end_idx)

