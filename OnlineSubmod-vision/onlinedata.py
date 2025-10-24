
import logging
# import wandb
import os
import os.path as osp
import sys
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ray import tune
from cords.selectionstrategies.helpers.ssl_lib.param_scheduler import scheduler as step_scheduler
from cords.utils.data.data_utils import WeightedSubset
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, AdaptiveRandomDataLoader, StochasticGreedyDataLoader,\
    CRAIGDataLoader, GradMatchDataLoader, OnlineSubmodDataLoader, RandomDataLoader, WeightedRandomDataLoader, MILODataLoader, SELCONDataLoader
from cords.utils.data.dataloader.SL.nonadaptive import FacLocDataLoader, MILOFixedDataLoader
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.models import *
from cords.utils.data.data_utils.collate import *
import pickle
from datetime import datetime
import submod
from generate_order import compute_dino_image_embeddings, load_feature_model
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tf
device = "cuda"
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x=None, y=None, z=None):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x = self.x[index] if self.x is not None else None
        y = self.y[index]
        z = self.z[index]
        if(self.x is None): return y, z
        return x, y, z
    
def calculate_mean_std(dataloader):
    """
    Calculate mean and standard deviation per channel (RGB) for images in a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader object containing image batches.
    
    Returns:
        mean (Tensor): Mean values per channel (shape: [channels]).
        std (Tensor): Standard deviation values per channel (shape: [channels]).
    """
    # Initialize placeholders for sum and sum of squares
    mean = torch.zeros(3)  # Assuming RGB images (3 channels)
    std = torch.zeros(3)
    total_pixels = 0  # Total number of pixels across all images
    
    for images, _ in dataloader:
        # Flatten height and width into the batch dimension
        batch_samples = images.size(0)  # Batch size
        total_pixels += batch_samples * images.size(2) * images.size(3)  # Total pixels
        
        # Compute mean and std for the batch
        mean += images.sum(dim=[0, 2, 3])  # Sum over batch, height, and width
        std += (images ** 2).sum(dim=[0, 2, 3])  # Sum of squares over batch, height, and width
    
    # Final mean and std calculations
    mean /= total_pixels
    std = torch.sqrt((std / total_pixels - mean) ** 2)  # std = sqrt(E[x^2] - (E[x])^2)
    
    return np.array(mean.tolist()), np.array(std.tolist())

def extract_features_old(dataname, dataloader, load_from=None, return_features=True):
    feature_model, feature_extractor = load_feature_model(device)
    if(return_features and load_from is not None and os.path.exists(load_from)):
        print(f"***Features exist, Loading presaved from {load_from}***")
        features, labels, images = None, None, None
        with open(load_from, 'rb') as f:
            features, labels, images = pickle.load(f)
        indices = torch.randperm(features.size()[0])
        return features[indices], labels[indices], images[indices]
    print(f"***Features dont exist, Creating new features for {load_from}***")
    features_list = []
    labels_list = []
    images_list = []
    t = 0
    # mean, std = calculate_mean_std(dataloader)
    # mean = torch.from_numpy(mean).to("cuda")
    # std = torch.from_numpy(std).to("cuda")
    
    def emb(images):
        return compute_dino_image_embeddings(images,device, 
                                                       model=feature_model, feature_extractor=feature_extractor, 
                                                       return_tensor=True)
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        ch3Images = images
        # Convert the input tensor to FloatTensor if it's of type ByteTensor
        if images.shape[1] == 1:
            ch3Images = torch.cat([images,images,images], dim=1)
            # features = torch.cat([features,features,features], dim=1)
            # labels = torch.cat([labels,labels,labels], dim=1)
        if images.dtype == torch.uint8:
            images = images.float()
        
        if(return_features): features_batch = emb(ch3Images)
        mean, std = None, None
        if dataname == "cifar10_normalised":
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))
            std = np.array([0.2023, 0.1994, 0.2010]).reshape((1, 3, 1, 1))
        if dataname == "cifar100":
            mean = np.array([0.5071, 0.4865, 0.4409]).reshape((1, 3, 1, 1))
            std = np.array([0.2673, 0.2564, 0.2762]).reshape((1, 3, 1, 1))
        if dataname == "tinyimagenet":
            mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
            std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        if dataname == "fashion-mnist":
            mean = np.array([0.1307,]).reshape((1, 3, 1, 1))
            std = np.array([0.3081,]).reshape((1, 3, 1, 1))
        if dataname == "svhn":
            mean = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))
            std = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))
        if mean is not None:
            mean = torch.from_numpy(mean).to("cuda")
            std = torch.from_numpy(std).to("cuda")
            images = (images - mean)/std
            images = images.float()
            
        if(return_features): features_list.append(features_batch)
        labels_list.extend(labels)
        images_list.append(images)

    # Concatenate all feature batches into a single tensor
    if(return_features): features = torch.cat(features_list, dim=0)
    labels = torch.tensor(labels_list)
    images = torch.cat(images_list, dim=0)
    if(not return_features): return labels, images
    print(f"***Saving features and labels***")
    with open(load_from, 'wb') as f:
        pickle.dump((features, labels, images), f)
    return features, labels, images

def are_datasets_equal(ds1, ds2) -> bool:
    if len(ds1) != len(ds2):
        print("Zero")
        return False

    # Compare the content of the datasets
    for idx in range(len(ds1)):
        item1 = ds1[idx]
        item2 = ds2[idx]
        
        # Check if the datasets return tuples
        if isinstance(item1, tuple) and isinstance(item2, tuple):
            if len(item1) != len(item2):
                return False
            # Compare each element in the tuple
            for sub_item1, sub_item2 in zip(item1, item2):
                if not torch.equal(sub_item1, sub_item2):
                    return False
        else:
            # Compare directly if not tuples
            if not torch.equal(item1, item2):
                return False

    # If everything matches, return True
    return True

def get_loaders(dataname, data_dir):
    print("DATA", dataname, data_dir)
    valset = None
    mean, std = None, None
    if(dataname == "mnist"):
        trainset, valset, testset, _, (mean, std)= gen_dataset(data_dir,"mnist","dss")
    elif(dataname == "tinyimagenet"):
        trainset, valset, testset, _, (mean, std) = gen_dataset(data_dir,"tinyimagenet","dss")
    elif(dataname == "fashion-mnist"):
        trainset, valset, testset, _, (mean, std)= gen_dataset(data_dir,"fashion-mnist","dss")
    elif(dataname == "stl10"):
        trainset, testset, _, _ = gen_dataset(data_dir,"stl10","dss")
    elif(dataname == "svhn"):
        # print("data", dataname)
        trainset, valset, testset, _, (mean, std)= gen_dataset(data_dir,"svhn","dss")
    elif(dataname == "cifar100"):
        # print("data", dataname)
        trainset, valset, testset, _, (mean, std)= gen_dataset(data_dir,"cifar100","dss")
    elif dataname == "cifar10":
        # Define data transforms
        trainset, valset, testset, _, (mean, std)= gen_dataset(data_dir,"cifar10","dss")
    else:
        raise
    return trainset, valset, testset, mean, std
