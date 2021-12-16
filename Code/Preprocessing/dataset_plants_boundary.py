import os
import torch
import numpy as np
import random
import skimage.segmentation
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image


def mask_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=Image.NEAREST), #T.InterpolationMode.NEAREST
         T.RandomHorizontalFlip(),
         T.RandomVerticalFlip(),
         T.ToTensor(),
         ])
    return transform


def image_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
         T.RandomHorizontalFlip(),
         T.RandomVerticalFlip(),
         T.ToTensor(),
         ])
    return transform


def general_transforms():
    random_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(), ])
    return random_transform


class CustomDataset(Dataset):
    def __init__(self, dir, transform, image_transform, mask_transform):
        self.directory = dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_transform = image_transform
        self.contents = os.listdir(dir)
        self.images = list(filter(lambda k: 'rgb' in k, self.contents))

        self.store_masks = []
        self.store_images = []

        for index, img in enumerate(self.images):
            img_path = os.path.join(self.directory, self.images[index])
            mask_path = os.path.join(self.directory, self.images[index].replace('rgb.png', 'label.png'))

            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask = np.array(skimage.segmentation.find_boundaries(mask, mode='thick'), dtype=np.float32)
            #mask = np.array(skimage.morphology.binary_dilation(mask), dtype = np.float32)

            self.store_masks.append(mask)
            self.store_images.append(image)

        print('Done Initiating Dataset')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.store_images[index]
        mask = self.store_masks[index]

        seed = np.random.randint(np.iinfo('int32').max)

        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)

            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        if self.mask_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)

            mask = self.mask_transform(mask)

        if self.image_transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)

            image = self.image_transform(image)

        return image, mask


if __name__ == '__main__':
    from Code.UNET import UNET

    HEIGHT, WIDTH = 400, 400
    image_directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'

    Plants = CustomDataset(image_directory, transform=None,
                           image_transform=image_train_transform(HEIGHT, WIDTH),
                           mask_transform=mask_train_transform(HEIGHT, WIDTH))

    # dataloader = DataLoader(Plants, batch_size = 4, shuffle = False)
    img_example, target_example = Plants.__getitem__(3)

    print(np.unique(target_example))

    input = img_example.unsqueeze(0)
    model = UNET()
    output = model(input)
    print(output.shape)

    # prediction = output.squeeze(0).detach().numpy().transpose(1,2,0)
    # thresh = 0.4
    # th, dst = threshold(prediction, thresh, 255, cv2.THRESH_BINARY);
    # plt.imshow(dst)

    fig = plt.figure(figsize=(10, 4))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('Image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.imshow(np.array(img_example.permute(1, 2, 0)))

    ax2.set_title('Boundaries')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    print(np.unique(np.array(target_example.permute(1, 2, 0).squeeze())))

    ax2.imshow(np.array(target_example.permute(1, 2, 0).squeeze()), cmap='gray')

    plt.savefig('Image_Boundary.png')
    plt.show()
    print('finished')

# print(f'Train dataset has {len(dataloader)} batches of size {4}')
