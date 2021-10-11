import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from UNET import UNET
from Preprocessing.dataset_plants_boundary import CustomDataset
from Preprocessing.plant_transforms import image_train_transform, mask_train_transform


def trainer():
    LEARNING_RATE = 0.001 #1e-3 empfohlen
    #lambda_1 = lambda epoch: 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 2
    HEIGHT, WIDTH = 100, 100
    IN_CHANNELS = 3  # RGB
    OUT_CHANNELS = 1 #output dimensions of embedding space

    torch.manual_seed(1)


    image_directory = '/Users/luisa/Documents/BA_Thesis/Datasets for Multiple Instance Seg/CVPPP2017_instances/training/A1/'

    Plants = CustomDataset(image_directory,
                                   transform=None,
                                   image_transform= image_train_transform(HEIGHT, WIDTH),
                                   mask_transform= mask_train_transform(HEIGHT, WIDTH)
                                   )

    train_set, val_set = torch.utils.data.random_split(Plants, [100, 28])

    dataloader = DataLoader(train_set, batch_size=8, shuffle=False)
    validation_loader = DataLoader(val_set, batch_size=8, shuffle = False)


    model_dir = '/Users/luisa/Documents/BA_Thesis/BoundaryPred UNet/Code/saved_models'
    model = UNET()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    #scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda_1)

    loss_function = nn.BCEWithLogitsLoss()
    #writer = SummaryWriter('runs/multi_runs')

    loss_statistic = np.array([])
    validation_loss_statistic = np.array([])

    #print(optimizer.state_dict())

    for i in range(0, EPOCHS):
        print('Entering Training Epoch {} out of {}'.format(i, EPOCHS))

        #model.train()
        running_loss = 0

        for images, targets in dataloader:
            optimizer.zero_grad()
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            #Predict
            preds = model(images)
            loss = loss_function(preds, targets)

            #Train
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss.item())

            #torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))



        #scheduler.step()

        loss_statistic = np.append(loss_statistic, running_loss)


        #Validation Loss
        with torch.no_grad():
            running_validation_loss = 0
            for images, targets in validation_loader:
                predictions = model(images)
                validation_loss = loss_function(predictions, targets)
                running_validation_loss += validation_loss

        validation_loss_statistic = np.append(validation_loss_statistic, validation_loss)


        #writer.add_scalar('Loss/training_loss of batch', running_loss, i)
        #writer.flush()


        #loss_output_epoch = train_function(dataloader, model, optimizer, loss_function, DEVICE)
        #torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(i)))

        print('')
        print('Completed {}/{} Epochs of Training'.format(i + 1, EPOCHS))

    torch.save(model, os.path.join(model_dir, 'epoch-{}.pt'.format(EPOCHS)))


    plt.title('Statistic for Train Loss and Validation Loss ')
    plt.scatter(np.linspace(1, EPOCHS, EPOCHS), loss_statistic, label = 'Train Loss')
    plt.scatter(np.linspace(1, EPOCHS, EPOCHS), validation_loss_statistic, label = 'Validation Loss')

    plt.grid()
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend(borderpad = True )
    plt.show()

if __name__ == '__main__':
    trainer()
