import torch
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import dice_loss
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")


from UNET import UNET
from Preprocessing.dataset_plants_boundary import CustomDataset, image_train_transform, mask_train_transform

def trainer():
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    LEARNING_RATE = 0.001 #1e-3 empfohlen
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device = ', DEVICE)
    EPOCHS = 2000
    HEIGHT = 512
    WIDTH = HEIGHT
    IN_CHANNELS = 3  # RGB
    OUT_CHANNELS = 1 #output dimensions of embedding space
    BATCH_SIZE = 4

    rel_image_dir = '~/Documents/BA_Thesis/CVPPP2017_instances/training/A1'
    image_directory = os.path.expanduser(rel_image_dir)

    #rel_val_dir = '~/Documents/BA_Thesis/CVPPP2017_instances/output/val/A1'
    #val_directory = os.path.expanduser(rel_val_dir)

    Plants = CustomDataset(image_directory,
                                   transform=None,
                                   image_transform= image_train_transform(HEIGHT, WIDTH),
                                   mask_transform=mask_train_transform(HEIGHT, WIDTH))

    random.seed(0)
    torch.manual_seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 20, 28])

    train_size = len(train_set)
    val_size = len(val_set)
    test_size = len(test_set)

    dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True)
    validation_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    relative_path = '~/Documents/BA_Thesis/BoundaryPred_UNet/Code/saved_models'
    model_dir = os.path.expanduser(relative_path)

    model = UNET().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_function = nn.BCEWithLogitsLoss() #nn.BCELoss() #dice_loss #nn.BCEWithLogitsLoss()

    loss_statistic = np.array([])
    validation_loss_statistic = np.array([])
    #map_location="cpu"
    #print(optimizer.state_dict())

    for i in tqdm(range(0, EPOCHS)):
        #print('Entering Training Epoch {} out of {}'.format(i, EPOCHS))
        model.train()
        running_loss = 0

        for images, targets in dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            #zero parameter gradients
            optimizer.zero_grad()

            #Predict/Forward
            preds = model(images)
            loss = loss_function(preds, targets)

            #Train
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            #print(loss.item())


        loss_statistic = np.append(loss_statistic, running_loss)

        if i in [0, 500, 1000, 1500, 2000]:
            #torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))
            prediction_out = preds.squeeze().cpu().detach().numpy()[0]
            plt.imsave('saved_images/Prediction-{}-{}.png'.format(EPOCHS, HEIGHT), prediction_out)


        #Validation Loss
        model.eval()
        with torch.no_grad():
            running_validation_loss = 0
            for images, targets in validation_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                predictions = model(images)
                validation_loss = loss_function(predictions, targets)
                running_validation_loss += validation_loss * images.size(0)

        validation_loss_statistic = np.append(validation_loss_statistic, running_validation_loss.cpu())

        #loss_output_epoch = train_function(dataloader, model, optimizer, loss_function, DEVICE)
        #torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))

        #print('')
        #print('Completed {}/{} Epochs of Training'.format(i + 1, EPOCHS))

    torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(EPOCHS)))

    plt.figure(figsize=(12, 9))
    plt.title('Statistic for Train Loss and Validation Loss for {}x{} images and Dice Loss, Epochs:{}'.format(HEIGHT, WIDTH, EPOCHS))

    plt.plot(np.linspace(1, EPOCHS, EPOCHS), loss_statistic/train_size, label = 'Train Loss')
    plt.plot(np.linspace(1+1, EPOCHS+1, EPOCHS), validation_loss_statistic/val_size, label = 'Validation Loss')
    plt.grid()
    plt.xlabel('Training Epoch')
    plt.ylabel('Training and Validation Loss')
    plt.yscale('log')
    plt.legend(borderpad = True )
    plt.savefig('saved_images/Statistic_boundary_loss{}.png'.format(EPOCHS))
    #plt.show()

if __name__ == '__main__':
    trainer()
