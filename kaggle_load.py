import os
import shutil
import pandas as pd
import glob
import json
#import cv2
#import plotly.express as px
import matplotlib.pyplot as plt
import torch
#import tensorflow as tf
from PIL import Image

import torch 
from torchvision import models, transforms 

def resnet_code(image_path):
    model = models.resnet50(pretrained=True) 
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    for param in model.fc.parameters():
        param.requires_grad = True

    model.eval() 
    # Load and preprocess an image 
    img = Image.open(image_path) 
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    img_tensor = preprocess(img) 
    # img_tensor = torch.unsqueeze(img_tensor, 0)
    


def main():
    #test 12500, dog 12499, cat 12499r
    cat_num = torch.randint(low=0, high=12499, size=(1,)).item()
    dog_num = torch.randint(low=0, high=12499, size=(1,)).item()
    path_cat = "/Users/stepansharkov/Downloads/cs330-project/kaggle_data/train/cat." + str(cat_num) + ".jpg"
    path_dog = "/Users/stepansharkov/Downloads/cs330-project/kaggle_data/train/dog." + str(dog_num) + ".jpg"
    random_nums_used = []
    for i in range (99):
        rannum = torch.randint(low=0, high=12500, size=(1,)).item()
        while rannum in random_nums_used:
            rannum = torch.randint(low=0, high=12500, size=(1,)).item()
        random_nums_used.append(rannum)
        path_test = "/Users/stepansharkov/Downloads/cs330-project/kaggle_data/test1/" + str(rannum) + ".jpg"
        resnet_code(path_test)
        # with open(path_test, "r+") as filehandle:
        #     print(path_test)

if __name__ == "__main__":
    main()