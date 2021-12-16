import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from UNET import UNET
from Preprocessing.dataset_plants_boundary import CustomDataset, image_train_transform, mask_train_transform

HEIGHT = 400
WIDTH = HEIGHT


rel_image_dir = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
image_directory = os.path.expanduser(rel_image_dir)

Plants = CustomDataset(image_directory,
                        transform=None,
                        image_transform= image_train_transform(HEIGHT, WIDTH),
                        mask_transform= mask_train_transform(HEIGHT, WIDTH))

image, mask = Plants.__getitem__(5)
image = image.unsqueeze(0)
relative_path = '~/Documents/BA_Thesis/BoundaryPred_UNet/Code/saved_models/epoch-100.pt'
model_dir = os.path.expanduser(relative_path)

loaded_model = torch.load(model_dir)

loaded_model.eval()
prediction = loaded_model(image).detach().numpy().squeeze()

print('Showing Prediction')
plt.imshow(prediction, cmap = 'gray')

#print(np.max(np.unique(prediction)))
#thresh = 0.1
#th, dst = threshold(prediction, thresh, 255, cv2.THRESH_BINARY);
#plt.imshow(dst)
plt.show()

