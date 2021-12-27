import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

from UNET import UNET
from Preprocessing.dataset_plants_boundary import CustomDataset, image_train_transform, mask_train_transform

HEIGHT = 512
WIDTH = HEIGHT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

rel_image_dir = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
image_directory = os.path.expanduser(rel_image_dir)

Plants = CustomDataset(image_directory,
                        transform=None,
                        image_transform= image_train_transform(HEIGHT, WIDTH),
                        mask_transform= mask_train_transform(HEIGHT, WIDTH))

random.seed(0)
torch.manual_seed(0)
train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])
image, mask = val_set.__getitem__(0)
#Plants.__getitem__(0)
image = image.unsqueeze(0)

image, mask = image.to(DEVICE), mask.to(DEVICE)
relative_path = '~/Documents/BA_Thesis/BoundaryPred_UNet/Code/saved_models/epoch-2000.pt'
loaded_model = UNET()
model_dir = os.path.expanduser(relative_path)

loaded_model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
loaded_model.eval()
loaded_model.to(DEVICE)

prediction = loaded_model(image).detach().numpy().squeeze()

print('Showing Prediction')
plt.yticks([])
plt.xticks([])
plt.imshow(prediction, cmap = 'gray')

#print(np.max(np.unique(prediction)))
#thresh = 0.1
#th, dst = threshold(prediction, thresh, 255, cv2.THRESH_BINARY);
#plt.imshow(dst)
plt.show()

