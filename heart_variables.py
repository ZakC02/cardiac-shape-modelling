import os
import torch


"""

This file contains the path variables used in the project : base_path and output_path
It adapts the paths to the environment (local or colab)

"""


isColab = False

# Define paths
base_path = 'training'
output_path = 'all_patient_images'
model_path = 'models/vae_model_99.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if isColab:
  """
  !pip install torchio
  !pip install umap-learn
  !pip install umap-learn[plot]
  !pip install bokeh
  """
  # à tester sur colab
  os.system('pip install torchio')
  os.system('pip install umap-learn')
  os.system('pip install umap-learn[plot]')
  os.system('pip install bokeh')

  
  from google.colab import drive
  drive.mount('/content/drive')

  # Put your own
  colab_prepath = '/content/drive/MyDrive/Ressources TélécomParis/IMA206 Photographie computationnelle   Méthodes par patchs/Projet de groupe/current_branch/'
  base_path = colab_prepath + base_path
  output_path = colab_prepath + output_path
  model_path = colab_prepath + model_path