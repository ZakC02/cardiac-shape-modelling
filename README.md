# Variational Auto-Encoders for Cardiac Shape Modeling

This repository contains the implementation and analysis of a Variational Auto-Encoder (VAE) model applied to cardiac shape modeling. This project was conducted as part of the IMA 206 course at Télécom Paris.

## Project Overview

The project focuses on developing a Variational Auto-Encoder (VAE) model to analyze and generate 3D cardiac images. The primary objectives include:

1. **Model Description**: 
   - The model is designed to process 3D cardiac images segmented into four zones: background, myocardium, and the right and left ventricles.
   - The dataset includes images of 150 patients with associated metadata that indicate cardiac conditions like dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), and others.

2. **Data Preprocessing**: 
   - Only images corresponding to the maximum dilation and contraction of the heart were retained.
   - The images were cropped to remove unnecessary background and resized to 128x128 pixels.

3. **Model Architecture**:
   - A β-VAE was implemented, featuring a convolutional encoder and decoder.
   - The encoder consists of convolutional layers followed by fully connected layers, while the decoder includes transposed convolutional layers.
   - ELU (Exponential Linear Unit) activations were used throughout the network.

4. **Model Evaluation**:
   - **Reconstruction Performance**: Evaluated using the Sørensen-Dice index.
   - **Generation Performance**: Assessed using the Fréchet Inception Distance (FID), comparing the distribution of real and generated images.

5. **Analysis of Latent Space**:
   - UMAP (Uniform Manifold Approximation and Projection) was used to verify overfitting and analyze outliers in the latent space.
   - Various parameters like the latent dimension, batch size, and β were optimized to enhance model performance.

## Key Results

- Optimal parameters for the model were found to be:
  - Latent dimension: 8
  - Batch size: 8
  - β value: 100
- The model successfully reconstructed and generated realistic 3D cardiac shapes, with a detailed analysis of how different parameters impacted performance.

## Conclusion and Future Work

The project demonstrates the potential of VAEs in cardiac shape modeling, with suggestions for further improvements:
- Excluding empty slices from the training data.
- Ensuring the consistency of the generated segmentations.
- Training a model that takes into account the 3D consistency of MRI scans.

## Contributors

- Yanis Aït El Cadi
- Zakaria Chahboune
- Clément Leroy
- Tanguy Schmidtlin

## Institution

This project was completed as part of the IMA 206 course at **Télécom Paris** in 2024.
