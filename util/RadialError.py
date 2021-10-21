import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# import torchvision.utils as vutils
# import json
import PIL
# import csv
# import scipy
# from PIL import Image

class generateDatasets(object):
    def __init__(self, PATH, datasetName = "TwoSourcesRdm", batch_size=40, num_workers=8, std_tr=0.25, s=512):
        self.bsize = batch_size
        self.nworkers = num_workers
        self.df_test = pd.read_csv(os.path.join(PATH, datasetName) + "/test.csv")
        self.df_train = pd.read_csv(os.path.join(PATH, datasetName) + "/train.csv")
        self.test = MyData(self.df_test, PATH, datasetName=datasetName, dset="test", std=0.0, s=s)
        self.train = MyData(self.df_train, PATH, datasetName=datasetName, dset="train",std= std_tr, s=s)

    def outputDatasets(self, typeSet = "test"):
        if typeSet == "test":
            return self.test, self.df_test
        elif typeSet == "train":
            return self.train, self.df_train

    def getWeights(self):
        wTest = np.zeros(self.df_test.Label.unique().size)
        for i in range(self.df_test.Label.size):
            wTest[int(self.df_test.Label[i])-1] += 1
        wTrain = np.zeros(self.df_train.Label.unique().size)
        for i in range(self.df_train.Label.size):
            wTrain[int(self.df_train.Label[i])-1] += 1
        if np.prod(wTest == self.df_test.Label.size/len(wTest)):
            print("Labels are balanced in test set")
        if np.prod(wTrain == self.df_train.Label.size/len(wTrain)):
            print("Labels are balanced in train set")
        return wTest, wTrain

    def getDataLoaders(self):
        trainloader = torch.utils.data.DataLoader(self.train, batch_size=self.bsize,
                    shuffle=True, num_workers=self.nworkers)
        testloader = torch.utils.data.DataLoader(self.test, batch_size=self.bsize,
                    shuffle=True, num_workers=self.nworkers)
        return trainloader, testloader

    def spinVsTemp(self):
        meanSpin_Te = np.zeros(10)
        Temp_Te = np.zeros(10)
        meanSpin_Tr = np.zeros(10)
        Temp_Tr = np.zeros(10)
        jj = 10
        ii_Te = int(self.test.__len__()/jj)
        ii_Tr = int(self.train.__len__()/jj)
        for j in range(jj):
            ms = 0
            mT = 0
            for i in range(ii_Te):
                ms += self.test[j*ii_Te + i][0].mean().item()
                mT += self.test[j*ii_Te + i][1].item()
            meanSpin_Te[j] = ms/ii_Te
            Temp_Te[j] = mT/ii_Te

            ms = 0
            mT = 0
            for i in range(ii_Tr):
                ms += self.train[j*ii_Tr + i][0].mean().item()
                mT += self.train[j*ii_Tr + i][1].item()
            meanSpin_Tr[j] = ms/ii_Tr
            Temp_Tr[j] = mT/ii_Tr

        T_Te = Temp_Te/10 + 1.8
        T_Tr = Temp_Tr/10 + 1.8
        plt.plot(T_Te, meanSpin_Te)
        plt.legend("Test")
        plt.xlabel("Temp")
        plt.ylabel("Mean Spin")
        plt.plot(T_Tr, meanSpin_Tr)
        plt.legend("Training")
        plt.show()

class MyData(Dataset):
    def __init__(self, df, PATH, datasetName="TwoSourcesRdm", dset = "test", std=0.25, s=512):
        self.s = s
        self.df = df
        self.path = os.path.join(PATH, datasetName, dset)
        self.t_noise = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize(s, interpolation=PIL.Image.NEAREST),
                               transforms.ToTensor(),
                               AddGaussianNoise(0., std),
                               ])
        self.t = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize(s, interpolation=PIL.Image.NEAREST),
                               transforms.ToTensor(),
                               #AddGaussianNoise(0., std),
                               ])
        self.fileNames = os.listdir(self.path)
        self.fileNames.sort()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        file = self.path + "/" + self.df.Cell[index]
        file2 = self.path + "/" + self.df.Field[index]
        image = self.load_image(file)
        # label = self.df.Label[index].astype(np.long) - 1
        label = self.load_image(file2, add_noise=False)
        return image, label

    def load_image(self, file_name, add_noise=True):
        x = np.loadtxt(file_name).astype(np.float32).reshape(self.s,self.s)
        if add_noise:
            image = self.t_noise(x)
        else:
            image = self.t(x)
        return image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def fetch_dataset_name(idx):
    name_number_dict = {18: 'EighteenSrcsRdm', 
                            14:'FourteenSrcsRdm', 
                            13:'ThirteenSrcsRdm',
                            19:'NineteenSrcsRdm', 
                            12:'TwelveSrcsRdm', 
                            17:'SeventeenSrcsRdm',
                            20:'TwentySrcsRdm', 
                            15:'FifteenSrcsRdm', 
                            16:'SixteenSrcsRdm',
                           11:'ElevenSrcsRdm',
                           1:'1SourcesRdm',
                           2:'2SourcesRdm',
                           3:'3SourcesRdm',
                           4:'4SourcesRdm',
                           5:'5SourcesRdm',
                           6:'6SourcesRdm',
                           7:'7SourcesRdm',
                       8:'8SourcesRdm',
                       9:'9SourcesRdm',
                       10:'10SourcesRdm'}
    return name_number_dict[idx]
    

def get_sources_centers(image):
    
    image = np.array(255*image, dtype=np.uint8)
    ret, binary = cv2.threshold(image, 1, 255, 0)
    
    
    
    # https://stackoverflow.com/a/55806272
    major = cv2.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    centers = []
    
    for c in contours:
        # moment of inertia of the countour
        M = cv2.moments(c)

        #x, y
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
        
        # cv2.circle(binary, (int(center_x), int(center_y)), 20, (255,255,255))

        centers.append((round(center_x), round(center_y)))
    # cv2.imshow('im', binary)
    return centers

def create_circular_mask(center, radius = 1, h=512, w=512):
    # https://newbedev.com/how-can-i-create-a-circular-mask-for-a-numpy-array
    
    Y, X = np.ogrid[:h,:w]
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def create_ring_mask(center, radius=1, dr=5, h=512, w=512):
    inner = create_circular_mask(center, radius = radius, h=h, w=w)
    
    outer = create_circular_mask(center, radius = radius+dr, h=h, w=w)
    
    mask = outer * np.where(inner, False, True)
    
    return mask

def create_nan_circular_mask(center, radius = 1, h=512, w=512):
    mask = create_circular_mask(center, radius = radius, h=h, w=w)
    
    return np.where(mask, 1.0, float('nan'))

# @torch.no_grad()
def per_center_MAE(src_img, target_img, pred_im, radius_list, 
                   h=512, w=512):
    
    centers = get_sources_centers(src_img)
    
    error_dict = {c: [] for c in centers}
    
    for c in centers:
        for r in radius_list:
            mask = create_nan_circular_mask(c, r, h=h, w=w)
            error = np.nanmean(mask*np.abs(target_img - pred_im))
            error_dict[c].append([r, error])
    return error_dict
    

def radial_differences(target, predicted, center, r=50):
    """
    Computes the difference of target - predicted in the x and y axis. The difference is computed from center_x - r
    to center_x + r (same for the y direction). If center +- r is outside the image it computes the difference to the 
    edge.

    Parameters
    ----------
    target : numpy.array 
        Target image of the neural net. Must be already detached, in cpu mode, converted to numpy and be 2D.
    predicted : numpy.array 
        Predicted image by the neural net. Must be already detached, in cpu mode, converted to numpy and be 2D.
    center : tupple / list
        (x, y) coordinates of the center.
    r : int, optional
        +- distance from the center the difference will be computed. The default is 50.

    Returns
    -------
    diff_x : numpy.array
        The difference of target-predicted in the x direction
    xlims : tupple
        The used min and max x in the difference calculation
    diff_y : numpy.array
        The difference of target-predicted in the y direction
    ylims : tupple
        The used min and max y in the difference calculation

    """
    c=center
    big_x = min(c[0]+r, target.shape[0])
    small_x = max(c[0]-r, 0)
    
    big_y = min(c[1]+r, target.shape[1])
    small_y = max(c[1]-r, 0)
    
    
    diff_x =  target[small_x:big_x, c[1]] - predicted[small_x:big_x, c[1]]
    
    diff_y = target[c[0], small_y:big_y] - predicted[c[0], small_y:big_y]
    
    xlims = (small_x, big_x)
    
    ylims = (small_y, big_y)
    
    return diff_x,  xlims, diff_y, ylims


def get_radial_differences(src_img, target_img, pred_img):
    
    centers = get_sources_centers(src_img[0,:,:].detach().cpu().numpy())
    
    error_dict = {c: {'difference in x direction': None, 'xlims': None, 
                      'difference in y direction': None, 'ylims': None} for c in centers}
    
    for c in centers:
        diff_x, xlims, diff_y, ylims = radial_differences(target_img[0,:,:].detach().cpu().numpy(), 
                                                          pred_img[0,:,:].detach().cpu().numpy(),
                                                         c)
        error_dict[c]['difference in x direction'] = diff_x
        error_dict[c]['xlims'] = xlims
        
        error_dict[c]['difference in y direction'] = diff_y
        
    return error_dict


if __name__ == '__main__':
    PATH = r'D:\Google Drive IU\phdStuff\AI-project-with-javier\diffusion project'
    device = torch.device("cpu")
    
    with torch.no_grad():
        _, testloader = generateDatasets(datasetName = 'sample-set', 
                                                           batch_size=1, 
                                                           num_workers=1, 
                                                           std_tr=0.0, s=512).getDataLoaders()
        b = next(iter(testloader))
        x = b[0].to(device)
        target = b[1].to(device)
        # print(x.shape)
        
        centers = get_sources_centers(x[0,0,:,:].detach().cpu().numpy())
        
        c = centers[0]
        
        predicted = np.random.uniform(size=x[0,0,:,:].shape)
        
        diff_x, xlims, diff_y, ylims = radial_differences(target[0,0,:,:].detach().cpu().numpy(), predicted, c)
        
        xdist = np.arange(xlims[0], xlims[1])
        ydist = np.arange(ylims[0], ylims[1])
        plt.figure()        
        plt.plot(xdist, diff_x)
        plt.figure()
        plt.plot(ydist, diff_y)
        




