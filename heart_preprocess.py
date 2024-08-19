import os
import torchio as tio
import shutil
import numpy as np
import torch
import cv2
from heart_variables import base_path, output_path



"""

This file creates a new folder called 'all_patient_images'
This folder saves individual cropped slices of the nifti files
Each file is named as follows:
    patient number, group, height, weight, frame number, slice number

"""




# Define paths
base_path = base_path
output_path = output_path

# Clear the output directory
if os.path.exists(output_path):
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

os.makedirs(output_path, exist_ok=True)

# Loop through each patient's directory
for patient_num in range(1, 101):
    patient_dir = os.path.join(base_path, f'patient{patient_num:03d}')
    info_file = os.path.join(patient_dir, 'Info.cfg')

    # Read the Info.cfg file
    with open(info_file, 'r') as f:
        info = {}
        for line in f:
            key, value = line.strip().split(': ')
            info[key] = value

    # Extract patient information
    ED = info['ED']
    ES = info['ES']
    group = info['Group']
    height = int(float(info['Height']))
    weight = int(float(info['Weight']))

    # Ensure proper formatting for ED and ES
    ED_str = f'{int(ED):02d}'
    ES_str = f'{int(ES):02d}'

    # Define the nifti file names
    nifti_files = [
        f'patient{patient_num:03d}_frame{ED_str}_gt.nii.gz',
        f'patient{patient_num:03d}_frame{ES_str}_gt.nii.gz'
    ]

    # Loop through nifti files and process
    for nifti_file in nifti_files:
        src_path = os.path.join(patient_dir, nifti_file)
        frame = ED_str if f'frame{ED_str}' in nifti_file else ES_str
        dst_filename_base = f'{patient_num:03d}_{group}_{height}_{weight}_{frame}'

        # Use torchio to load the nifti file
        image = tio.LabelMap(src_path)
        image_data = image.data.squeeze().numpy()

        # Calculate the largest bounding box across all slices
        max_x, max_y, max_w, max_h = 0, 0, 0, 0
        for idx in range(image_data.shape[-1]):
            slice_data = image_data[..., idx]
            contours, _ = cv2.findContours(slice_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                max_hw = max(w, h)  # Size of the square
                if max_hw * max_hw > max_w * max_h:
                    max_x, max_y, max_w, max_h = x, y, max_hw, max_hw

        # Apply padding to the bounding box
        padding = 10
        x_start = max(max_x - padding, 0)
        y_start = max(max_y - padding, 0)
        x_end = min(max_x + max_w + padding, image_data.shape[1])
        y_end = min(max_y + max_h + padding, image_data.shape[0])

        # Loop through each slice, crop using the largest bounding box, and save
        for idx in range(image_data.shape[-1]):
            slice_data = image_data[..., idx]
            cropped_image = slice_data[y_start:y_end, x_start:x_end]
            slice_filename = f'{dst_filename_base}_{idx + 1}.nii.gz'
            slice_path = os.path.join(output_path, slice_filename)

            # Convert back to 4D tensor (adding batch and channel dimensions)
            cropped_image_tensor = torch.tensor(cropped_image).unsqueeze(0).unsqueeze(0)

            # Save the individual cropped slice
            slice_image = tio.LabelMap(tensor=cropped_image_tensor, affine=image.affine)
            slice_image.save(slice_path)

print("All files have been processed, and individual cropped slices have been saved.")

