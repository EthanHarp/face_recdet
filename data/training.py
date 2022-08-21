#inspo = https://github.com/timesler/facenet-pytorch/blob/master/examples/finetune.ipynb
#if model doesn't work, try removing squeeze in getitem and revert to old image opener
#REMOVE UNUSED IMPORTS
from PIL import Image
import pandas as pd
from torchvision.io import read_image
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

# class_to_idx = {
#     1: "Ethan Harpster"
# }


#Define transform
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization,
])

#Define Custom Dataset
class PersonalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels, trans):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels)
        self.transform = trans

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        #image = read_image(img_path)
        image = Image.open(img_path)
        image = self.transform(image)
        label = self.labels.iloc[idx, 1]
        image = image.squeeze(1)
        return image, label



#Method for finetuning model
def train_model(model, train_loader, val_loader, epochs=25):
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    model.eval()
    training.pass_epoch(model, loss_fn, val_loader, batch_metrics=metrics, show_running=True, device=device, writer=writer)

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        model.train()
        training.pass_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        model.eval()
        training.pass_epoch(
            model, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()
    return model

#define file paths
img_dir = '/YOUR_DIRECTORY/face_recdet/data/dataset'
label_file = '/YOUR_DIRECTORY/face_recdet/data/labels.csv'
#Make dataset of all pics
all_data = PersonalDataset(img_dir, label_file, trans)
#split dataset into training and validation sets
train_size = 72
val_size = len(all_data) - train_size
train_data, val_data = torch.utils.data.random_split(all_data, [train_size, val_size])
#Create training dataloader and validation dataloader
train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=12, shuffle=True)
#Grab gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Running on device: {}'.format(device))
#define model
model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
#    num_classes=len(all_data.class_to_idx)
    num_classes=2
).to(device)

#Train the model


optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])


loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

model = train_model(model, train_dataloader, val_dataloader)


#Save the model
torch.save(model, 'CustomModel.pyh')