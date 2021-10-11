import os
from PIL import Image
import numpy as np
import skimage.segmentation
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from plant_transforms import mask_train_transform, image_train_transform



class CustomDataset(Dataset):
    def __init__(self, dir, transform, image_transform, mask_transform):
        self.directory = dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_transform = image_transform
        self.contents = os.listdir(dir)
        self.images = list(filter(lambda k: 'rgb' in k,self.contents ))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.directory, self.images[index])
        mask_path = os.path.join(self.directory, self.images[index].replace('rgb.png', 'fg.png'))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype= np.float32)

        kernel = np.ones((5, 5), np.uint8)
        mask = np.array(skimage.segmentation.find_boundaries(mask), dtype=np.float32)
        #mask = np.array(mask/255, dtype=np.int64)

        if self.image_transform is not None:
            print('start image transform')
            image = self.image_transform(image)

        if self.mask_transform is not None:
            print('start mask transform')
            mask = self.mask_transform(mask)

        return image, mask


image_directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2015_LCC_training_data/A1'

"""
contents = os.listdir(image_directory)
images = list(filter(lambda k: 'rgb' in k,contents ))
masks = list(filter(lambda k: 'fg' in k, contents))

path_name = os.path.join(image_directory, images[index].replace('rgb.png', 'fg.png'))
"""


#Plants = CustomDataset(image_directory)

#dataloader = DataLoader(Plants, batch_size = 4, shuffle = False)
#img_example, mask_example = Plants.__getitem__(3)

#print(img_example.shape)

#plt.subplot(1,2,1)
#plt.title('Image')
#plt.imshow(img_example)

#plt.subplot(1,2,2)
#plt.title('Mask')
#plt.imshow(mask_example)

#plt.show()

#print(f'Train dataset has {len(dataloader)} batches of size {4}')
