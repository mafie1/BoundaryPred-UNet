import matplotlib.pyplot as plt
import torch
import random
import os
from Preprocessing.dataset_plants_boundary import CustomDataset, image_train_transform, mask_train_transform

random.seed(0)
torch.manual_seed(0)

HEIGHT, WIDTH = 100, 100

rel_image_dir = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1'
image_directory = os.path.expanduser(rel_image_dir)


Plants = CustomDataset(dir = image_directory,
                               transform= None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

"""Insert Train_test_split"""
train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 20, 28])


img_example, boundary_example = val_set.__getitem__(1)
img_example = img_example.unsqueeze(0)

loaded_model = torch.load('/Users/luisaneubauer/Documents/BA_Thesis/BoundaryPred_UNet/Code/saved_models/epoch-10.pt')
loaded_model.eval()

boundary_pred = loaded_model(img_example).squeeze(0).squeeze(0).detach().numpy()

plt.yticks([])
plt.xticks([])
plt.title('Prediction on Validation Image')
#plt.imshow(boundary_pred, cmap = 'gray', interpolation = 'nearest')
#plt.imshow(img_example.squeeze(0).permute(1,2,0))
plt.imshow(boundary_example.squeeze(0), interpolation = 'nearest')
plt.show()