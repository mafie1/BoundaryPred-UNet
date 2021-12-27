import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from skimage.segmentation import slic, join_segmentations, watershed

sys.path.insert(0, os.path.expanduser('~/Documents/BA_Thesis/Image_Embedding_Net/Code/Preprocessing'))
sys.path.insert(1, os.path.expanduser('~/Documents/BA_Thesis/Image_Embedding_Net/Code/Postprocessing'))
sys.path.insert(2, os.path.expanduser('~/Documents/BA_Thesis/Image_Embedding_Net/Code/'))
sys.path.insert(3, os.path.expanduser('~/Documents/BA_Thesis/Mean_Shift/Code_MS_Shift'))
sys.path.insert(2, os.path.expanduser('~/Documents/BA_Thesis/BoundaryPred_UNet/Code'))

from metrics import get_SBD, counting_score
from model import UNet_spoco_new
from UNET import UNET
from utils_cluster import load_model, load_image_mask, cluster_hdbscan, cluster_dbscan, cluster_agglo, cluster_ms, load_dataset
print('Imports successfully completed')


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
E_DIM = 8
Epoch = 2000

#Load image and ground truth
torch.manual_seed(0)
random.seed(0)

val_set = load_dataset(mode='val', height=512)
image, mask = val_set.__getitem__(0)

#load embedding model
model_path = os.path.expanduser('~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/full_UNet/run-dim{}-height512-epochs3000/epoch-{}-dim{}-s512.pt'.format(E_DIM, Epoch, E_DIM))
loaded_model = UNet_spoco_new(in_channels=3, out_channels=E_DIM)
loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
loaded_model = loaded_model.eval()
print('Successfully loaded model')

#predict mask
prediction_emb = loaded_model(image.unsqueeze(0)).detach().numpy().squeeze()
print(prediction_emb.shape)
print('Predicted Embedding Successfully')

prediction_mask = cluster_ms(prediction_emb, bandwidth = 2.2)-1.
print('Predicted Mask Successfully')

#load boundary net
model_dir_bnd = os.path.expanduser('~/Documents/BA_Thesis/BoundaryPred_UNet/Code/saved_models/epoch-2000.pt')
model_bnd = UNET()
model_bnd.load_state_dict(torch.load(model_dir_bnd, map_location=torch.device('cpu')))
model_bnd.eval()

#predict boundaries
prediction_bnd = model_bnd(image.unsqueeze(0)).detach().numpy().squeeze()
prediction_bnd = (prediction_bnd-np.min(prediction_bnd))/(np.max(prediction_bnd)-np.min(prediction_bnd))
print('Boundaries Predicted Successfully')

mask = mask.squeeze(0)

fig, ax = plt.subplots(2,2)

for x in range(0,2):
    for y in range(0,2):
        ax[x,y].set_xticks([])
        ax[x,y].set_yticks([])


ax[0,0].imshow(image.permute(1,2,0))
ax[0,0].set_title('Original Image')

ax[0,1].imshow(mask)
ax[0,1].set_title('Ground Truth Mask')

ax[1,0].imshow(prediction_bnd, cmap='gray')
ax[1,0].set_title('Predicted Borders')

ax[1,1].imshow(prediction_mask)
ax[1,1].set_title('Predicted Mask')

fig.show()


#get DICE score between every instance in mask and every instance in predicted mask.
I = len(np.unique(mask))
J = len(np.unique(prediction_mask))

print('Instances in GT Mask:', I)
print('Instances in Predicted Mask', J)


overlaps = np.empty((I, J))
DICE = np.empty((I,J))

for i, instance_i in enumerate(np.unique(mask)):
    label_i = np.where(mask == instance_i, 1, 0)

    for j, instance_j in  enumerate(np.unique(prediction_mask)):
        label_j = np.where(prediction_mask == instance_j, 1, 0)

        overlap_ij = label_i * label_j
        overlap_pixels_ij = np.sum(overlap_ij)

        overlaps[i][j] = overlap_pixels_ij
        DICE[i][j] = 2*overlap_pixels_ij /( np.sum(label_j) + np.sum(label_i) )


"""Thresholding overlaps with DICE less than T =  0.1"""
T = 0.1
df = pd.DataFrame(data=DICE)
df = df[df > T]
print(df.head(17))
print(df.count(axis = 0))


sorted_ij = np.argsort(DICE, axis = 0) #sort within first axis

max_sorted = np.ndarray.max(sorted_ij, axis = 1)
print(max_sorted.shape)
print(max_sorted)

#return biggest overlap for first instance in GT mask [0]:

"""Example for 2. and 14. """
label_i_0 = np.where(mask == 15., 1, 0) # [1,13], [2, 14], [4,15]
label_j_16 = np.where(prediction_mask == 12., 1, 0) #

plt.imshow(label_i_0, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('Label_i_0 ( Ground Truth)')
plt.show()

plt.imshow(label_j_16, cmap = 'gray')
plt.title('Predicted')
plt.xticks([])
plt.yticks([])
plt.show()

#union = (label_j_16 + label_i_0)/ (label_j_16 + label_i_0)
union = np.logical_or(label_i_0, label_j_16)
union_numeric = np.where(union, 1, 0)

plt.imshow(union_numeri, cmap = 'gray')
plt.title('Union')
plt.show()

plt.imshow(prediction_bnd * union, cmap = 'gray')
plt.show()

"""Sum background pixels that lie within union"""
bng_pixel = np.sum(union*prediction_bnd)

#print(np.amax(union*prediction_bnd))
#print(np.amin(union*prediction_bnd))


#print(bng_pixel/union)




















