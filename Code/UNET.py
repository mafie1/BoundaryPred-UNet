import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),  #in the Unet from 2015, there is no padding
            nn.BatchNorm2d(num_features = out_channels), #should I use batchnorm? What does it do? Not in orignal paper but most implementations use it. Why? I read something about 'gradient explosion'. Anyway, probably doesn't hurt using it.
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(num_features = out_channels),#try out GroupNorm as alternative
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module): #requires input of shape [3,1,height, width]
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256]):#features is f_map in spoco 512

        super(UNET, self).__init__()

        self.ups = nn.ModuleList() #empty encoder modules
        self.downs = nn.ModuleList() #empty decoder modules
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #MaxPool layers after each double convolutional layer, stride 2 for downsampling

        # Down part of UNET (is encoder in spoco)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature


        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)  # bottom of U-net 1024 --> 512 features to 1024 features; could also be integrated into the Down part of UNet

        # Up part of UNET (is decoder in spoco)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,)) #here: opposite to the maxpool2d operation in the encoder --> upsampling,
            # Conv_Transpose2d halfes the number of channels and doubles height/width of image
            self.ups.append(DoubleConv(feature*2, feature))


        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) #final convolution; gives out segmentation_map, that is 2 output channels, 1x1 convolution

        #self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1) in spoco

    def forward(self, x):

        skip_connections = [] #array for the concatenation

        for down in self.downs:
            x = down(x)
            skip_connections.append(x) #save features before maxpooling
            x = self.pool(x)
            #print(x.shape)
        x = self.bottleneck(x) #now at 1024 features

        skip_connections = skip_connections[::-1] #reverse concatenation list for decoder path

        #print(len(self.ups))

        for idx in range(0, len(self.ups), 2): #take only every second element because self.ups contains conolutions as well as pooling layers
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:]) #resize if necessary; otherwise concatenation not possible

                #print(skip_connection.shape[:])

                #skip_connection = TF.resize(skip_connection, size = x.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x) #torch.sigmoid(self.final_cov(x))






#______________________________
def test():
    #x = torch.randn((3, 1, 218, 224))  #use multiples of 16 = 2^4 to avoid resizing.
    x = torch.randn((3, 1, 256, 256))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)

    print(preds.shape, x.shape)
    print(preds)
    assert preds.shape == x.shape #only for padding = 1




#sample_image = torch.from_numpy(sample_image)
#print(sample_image.shape)

#sample_image = torch.unsqueeze(sample_image,0)

#sample_image = torch.reshape(sample_image, (3, 1, 1280, 1918))


def image_to_input_tensor(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    height, width, channels = image.shape

    torch_image = torch.from_numpy(image).float()
    torch_image = torch.unsqueeze(torch_image, 3)
    torch_image = torch.reshape(torch_image, (1, channels, height, width))
    #print(torch_image.shape)


def output_to_image(output):
    image_out = torch.squeeze(output, 0)
    image_out = TF.to_pil_image(image_out)
    image_out.show()


def test2():
    sample_image = np.array(Image.open('ara2012_plant117_rgb.png').convert("RGB"))
    sample_image = torch.from_numpy(sample_image).float()
    sample_image = torch.unsqueeze(sample_image, 3)
    print(sample_image.shape)
    sample_image = torch.reshape(sample_image, (1, 3, 447, 456))
    #sample_image = TF.resize(sample_image, size=(1, 3, 100, 100))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(sample_image)

    with torch.no_grad():

        print(preds.shape)
        preds = torch.squeeze(preds, 0)
        preds = convert_to_numpy(preds)
        plt.imshow(preds)
        plt.show()


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays
    Args:
        inputs (iteable of torch.Tensor): torch tensor
    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)

#if __name__ == "__main__":
 #   test()


#print(sample_image.shape)

#sample_image = torch.reshape(sample_image, (1, 1280, 1918, 3))


#print(torch.randn((3, 1, 256, 256)).shape)

#sample_image = torch.squeeze(sample_image, 0)
#print(sample_image.shape)
#plt.imshow(sample_image)
#plt.show()

#def image_to_4dtensor(image):
    #image = torch.from_numpy((image))
