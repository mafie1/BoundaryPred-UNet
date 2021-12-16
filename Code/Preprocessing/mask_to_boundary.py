import numpy as np
import skimage.segmentation
import matplotlib.pyplot as plt
#from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple
from dataset_plants_boundary import CustomDataset, image_train_transform, mask_train_transform

HEIGHT, WIDTH = 256, 256

directory = '/Users/luisa/Documents/BA_Thesis/CVPPP2017_instances/training/A1/'
Plants = CustomDataset(dir = directory,
                               transform= None,
                               image_transform= image_train_transform(HEIGHT, WIDTH),
                               mask_transform= mask_train_transform(HEIGHT, WIDTH))

img_example, boundary_example = Plants.__getitem__(2)

print(img_example.type())
print(boundary_example.type())

#boundaries = skimage.segmentation.find_boundaries(mask_example).transpose(1,2,0)
#diluted_bound = cv2.dilate(boundaries, kernel, iterations=1).transpose(1,2,0)
#mask_example = mask_example.detach().numpy().transpose(1,2,0)
img_example = img_example.detach().numpy().transpose(1,2,0)
boundary_example = boundary_example.detach().numpy().transpose(1,2,0)
#boundary_example = boundary_example.detach().numpy().transpose(1,2,0)

fig = plt.figure()
fig.add_subplot(1, 2, 2)
plt.title('Boundaries')
plt.axis('off')
plt.imshow(np.rot90(boundary_example,2))

#fig.add_subplot(1, 3, 2)
#plt.title('Mask')
#plt.axis('off')
#plt.imshow(np.rot90(mask_example,2), cmap = 'hot')

fig.add_subplot(1, 2, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(np.rot90(img_example,2))

#fig.add_subplot(1, 4, 4)
#plt.title('Diluted Boundaries')
#plt.axis('off')
#plt.imshow(np.rot90(diluted_bound), 2)

#plt.savefig('Mask_and_boundary_noise.png')
plt.show(block=True)



