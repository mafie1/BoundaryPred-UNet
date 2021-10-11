import os
from PIL import Image
import numpy as np
import skimage.segmentation
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



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
            #print('start image transform')
            image = self.image_transform(image)

        if self.mask_transform is not None:
            #print('start mask transform')
            mask = self.mask_transform(mask)

        return image, mask



if __name__ == '__main__':
    from plant_transforms import mask_train_transform, image_train_transform
    from UNET import UNET
    from cv2 import threshold
    import cv2
    HEIGHT, WIDTH = 128, 128
    image_directory = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/'
    Plants = CustomDataset(image_directory, transform = None,
                           image_transform = image_train_transform(HEIGHT, WIDTH),
                           mask_transform = mask_train_transform(HEIGHT, WIDTH))

    dataloader = DataLoader(Plants, batch_size = 4, shuffle = False)
    img_example, target_example = Plants.__getitem__(3)

    print(img_example.shape)
    print(target_example.shape)

    input = img_example.unsqueeze(0)
    model = UNET()
    output = model(input)
    print(output.shape)

    prediction = output.squeeze(0).detach().numpy().transpose(1,2,0)
    thresh = 0.2
    th, dst = threshold(prediction, thresh, 255, cv2.THRESH_BINARY);

    plt.imshow(dst)
    plt.show()
    print('finished')

#plt.subplot(1,2,1)
#plt.title('Image')
#plt.imshow(img_example)

#plt.subplot(1,2,2)
#plt.title('Mask')
#plt.imshow(mask_example)

#plt.show()

#print(f'Train dataset has {len(dataloader)} batches of size {4}')
