import time
from tqdm import tqdm
import os
print('hello')


import os
#os.system('pip install torchio')

from heart_display import load_images, display_images
from heart_variables import output_path

start_idx = 1
end_idx = 100
#display_images(output_path, start_idx, end_idx)

from heart_variables import device
print(type(device))
