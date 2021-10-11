import torch
import torchvision.transforms as T
from PIL import Image


def mask_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = Image.NEAREST),
         T.ToTensor()])
    return transform


def image_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = T.Compose(
        [T.ToPILImage(),
         T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
         T.ToTensor(),
         #T.Normalize(
          #   mean=[0.485, 0.456, 0.406],
           #  std=[0.229, 0.224, 0.225])
         ])
    return transform

