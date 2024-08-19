import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode

from heart_variables import output_path
from heart_display import load_images


class HeartDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()  # on doit garder des int 
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        transformed_image = self.transform(image)*255  # A changer, on doit déja avoir des int 
        transformed_image = transformed_image.long() # à voir
        one_hot_image = F.one_hot(transformed_image, num_classes=4).permute(0, 3, 1, 2).squeeze()
        return one_hot_image.float()

# Load the images
# images, names = load_images(output_path)

# heart_dataset = HeartDataset(images)
# dataloader = DataLoader(heart_dataset, batch_size=32, shuffle=True)

# for batch in dataloader:
#     print(batch.shape)  # Should print: torch.Size([32, 4, 128, 128])
#     img = torch.permute(batch[10],(1,2,0))
#     plt.imshow(np.argmax(img, axis=2))
#     print(np.sum(np.argmax(img, axis=2).numpy()))
#     img = batch.numpy()
#     break