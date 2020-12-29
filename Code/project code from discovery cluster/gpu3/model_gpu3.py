## This script file is for Model Parallelism

# Imports

import time
from colored import fg, attr
from sklearn.metrics import accuracy_score, f1_score
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as model
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_auc_score
import cv2
import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensor 
from PIL import Image, ImageFile
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
from skimage.io import imread
import sklearn
from sklearn.model_selection import train_test_split
from skimage import io
from torch import optim
from torch import FloatTensor, LongTensor, DoubleTensor
from albumentations import Normalize, VerticalFlip, HorizontalFlip, Compose
from torch.utils.data.sampler import WeightedRandomSampler
import warnings
warnings.filterwarnings("ignore")


# train dataset

train_df = pd.read_csv("/home/dabas.a/project/train.csv")
print(train_df.head())


# create folds

df = train_df
df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values
kf = sklearn.model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

df_updated = df
# df.to_csv("train_folds.csv", index=False)


## 1 class

import cv2
class MyDataSet(torch.utils.data.Dataset):

    def __init__(self, image_path, targets, transforms=None):
        self.image_path = image_path
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = self.image_path[item]
        targets = self.targets[item]
        img = Image.open(image)
        img = np.array(img)
#         print(img.shape)
        img = cv2.imread(image,cv2.IMREAD_COLOR)
#         print(image)
#         print(targets)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        
        if self.transforms is not None:
            sample = {'image':img}
            sample = self.transforms(**sample)
            img = sample['image']
            #img = self.transforms(img)
            #print(img.shape)
        else:
            img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float), torch.tensor(targets, dtype=torch.float)
        
        
        
### class 2
df = df
def train_fold(fold):
    training_path = "/scratch/dabas.a/data/jpeg/train_resize"
    

    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    train_images = list(df_train.image_name)
    train_images = [os.path.join(training_path,i+'.jpg') for i in train_images]
    train_targets = df_train.target.values

    valid_images = list(df_valid.image_name)
    valid_images = [os.path.join(training_path,i+'.jpg') for i in valid_images]
    valid_targets = df_valid.target.values

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RGBShift(r_shift_limit=40),
        A.MultiplicativeNoise(p=1.0),
        A.RandomBrightness(0.1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensor()
    ])

    valid_transform = A.Compose([
        A.RandomRotate90(),
        A.RandomBrightness(0.1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensor()
    ])

    train_dataset = MyDataSet(train_images, train_targets, train_transform)
    valid_dataset = MyDataSet(valid_images, valid_targets, valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=256,shuffle=True,sampler=None)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False,sampler=None)

    return train_loader, valid_loader,valid_targets
    
    
# class 3
class Resnet_152(nn.Module):
    def __init__(self,model):
        super(Resnet_152, self).__init__()
        self.model = model
        self.ext = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100,1)
        )

    def forward(self,images):
        out = self.model(images)
        out = self.ext(out)
        return out
        
        
# cuda details

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# how many GPU's do we have
print("The number of GPU's we have are : ", torch.cuda.device_count() )



## class 4 

import torchvision.models as model

model = model.resnet152(pretrained=True)

for param in model.parameters():
    param.requires_grad=False


in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 100)


mymodel = Resnet_152(model)

## Run this model on Multiple GPU
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  mymodel = nn.DataParallel(mymodel)

# model to GPU's
mymodel = mymodel.to(device)


criteria = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(mymodel.parameters(), lr=0.0001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max',verbose=True)

#es = EarlyStopping(patience=5, mode="max")

no_of_folds = 1 

for i in range(no_of_folds):
    print('Fold={}'.format(i))
    tr_loader, val_loader,targets = train_fold(i+1)
    epochs = 10
    
    acc = {}
    loss_score = {}
    time_rec = {}
    
    
    for j in range(epochs):
                
        loss_arr = []
        mymodel.train()
        
        # starting the time
        start_time = time.time()
        
        #training data 
        
        for data in tr_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            del data
            optimizer.zero_grad()
            output = mymodel(images)
            #print(output)
            loss = criteria(output, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
            del images,labels
        print("epoch={},loss={}".format(j, sum(loss_arr)/len(loss_arr)))
        
        # save the details for current epoch 
        print("time for epoch", time.time() - start_time)
        time_rec[j] = time.time() - start_time
        loss_score[j] = sum(loss_arr)/len(loss_arr)
        
        mymodel.eval()
        final_predictions = []
        
        for val_data in val_loader:
            val_images, val_labels = val_data
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            del  val_data
            with torch.no_grad():
                val_output = mymodel(val_images)
                #proba, pred = torch.max(val_output.data, 1)
                #print(val_output)
                val_output = torch.sigmoid(val_output)
                pred = val_output.cpu()
                #final_predictions.extend(pred)
                final_predictions.append(pred)
                del val_images, val_labels
                
        
#       predictions = np.array(final_predictions)

        predictions = np.vstack(final_predictions).ravel()

        k = roc_auc_score(targets, predictions)
        #l=accuracy_score(targets, predictions)
        print('val_auc_acore={}'.format(k))
        
        # accuracy and f1 score plot
        acc[j] = k
        #f1score[j] = f1_score(targets, predictions)
        
        scheduler.step(k)
        

    with open('plot_acc.csv', 'w') as f:
      for key in acc.keys():
        f.write("%s,%s\n"%(key,acc[key]))
    
    with open('plot_loss.csv', 'w') as f:
      for key in loss_score.keys():
        f.write("%s,%s\n"%(key,loss_score[key]))
     
    
    with open('plot_time.csv', 'w') as f:
      for key in time_rec.keys():
        f.write("%s,%s\n"%(key,time_rec[key]))
    
          
    torch.save(mymodel.state_dict(), 'resnet152_gpu_4_{}.pth'.format(i))   