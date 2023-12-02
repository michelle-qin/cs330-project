import matplotlib.pyplot as plt
import torch
from torch import optim
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from clip_retrieval.clip_client import ClipClient
import requests
from PIL import Image
import io
import numpy as np

STRATEGY = "RANDOM"
# STRATEGY = "BASELINE"
# STRATEGY = "TEXT_RETRIEVEL"
# STRATEGY = "SEMANTIC_NEAREST_NEIGHBOR"
# STRATEGY = "CONTENT"


def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    # If a GPU is available, move the model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over the training data
        for inputs, labels in train_loader:
            # Move inputs and labels to the same device as the model
            inputs, labels = inputs.unsqueeze(dim=0).to(device), torch.tensor(
                labels
            ).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(preds, labels)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / len(train_loader)

        print(
            f"Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f} Train_Acc: {epoch_acc:.4f}"
        )

    return model


def eval_model(model, test_loader):
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.unsqueeze(dim=0), torch.tensor(labels)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(preds, labels)
            print(preds == labels)
            running_corrects += torch.sum(preds == labels)
    total_acc = running_corrects.double() / len(test_loader)
    print(f"Eval Accuracy: {total_acc}")


def process_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(image)  # (3, 224, 224)
    return img_tensor

def text_supplement_with_laion(train_dict, num_supplement=20):
    # MICHELLE

def semantic_NN_supplement_with_laion(train_dict, num_supplement=20):
    # KATE
    
def content_supplement_with_laion(train_dict, num_supplement=20):
    # STEPHAN

def random_supplement_with_laion(train_dict, num_supplement=20):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=500,
    )
    supplement_dict = {}
    for pet_name in train_dict.keys():
        pet_images = client.query(text="an image of a " + pet_name)
        num_pet_images = len(pet_images)
        print("num_pet_images", num_pet_images)

        random_nums_used = []
        supplement_dict[pet_name] = []
        while len(supplement_dict[pet_name]) < num_supplement:
            # GET A RANDOM IMAGE
            rannum = torch.randint(low=0, high=num_pet_images - 1, size=(1,)).item()
            while rannum in random_nums_used:
                rannum = torch.randint(low=0, high=num_pet_images - 1, size=(1,)).item()
            random_nums_used.append(rannum)

            image_path = pet_images[rannum]["url"]
            print("IMAGE PATH: ", image_path)
            try:
                response = requests.get(image_path)
                if response.status_code == 200:
                    try:
                        image = Image.open(io.BytesIO(response.content))
                        image_array = process_image(image)
                        supplement_dict[pet_name].append(
                            (image_array, train_dict[pet_name])
                        )
                    except error as e:
                        print("Issue with getting image: ", e)
            except requests.exceptions.RequestException as e:
                print("FAIL")

    return supplement_dict


def main():
    # CREATE INITIAL DATA
    # TRAIN DATA
    # test 12500, dog 12499, cat 12499r
    cat_num = torch.randint(low=0, high=12499, size=(1,)).item()
    dog_num = torch.randint(low=0, high=12499, size=(1,)).item()

    img_cat = Image.open("kaggle_data/train/cat." + str(cat_num) + ".jpg")
    img_dog = Image.open("kaggle_data/train/dog." + str(dog_num) + ".jpg")
    train_cat = process_image(img_cat)
    train_dog = process_image(img_dog)

    train_data = [train_cat, train_dog]
    train_dict = {"cat": [0], "dog": [1]}
    train_labels = [[0], [1]]
    train_loader = [(train_data[i], train_labels[i]) for i in range(len(train_data))]

    # SUPPLEMENT DATA
    if STRATEGY == "BASELINE": 
        pass
    else:
        if STRATEGY == "RANDOM":
            supplement_data = random_supplement_with_laion(train_dict)
        elif STRATEGY == "TEXT_RETRIEVAL":
            # MICHELLE TO DO
            pass
        elif STRATEGY == "SEMANTIC_NEAREST_NEIGHBOR":
            # KATE TO DO
            pass
        elif STRATEGY == "CONTENT":
            # STEPHAN TO DO
            pass
        for pet_name in train_dict.keys():
            train_loader.extend(supplement_data[pet_name])
        

    # TEST DATA
    random_nums_used = [cat_num, dog_num]
    test_loader = []
    for i in range(99):
        rannum = torch.randint(low=0, high=12500, size=(1,)).item()
        while rannum in random_nums_used:
            rannum = torch.randint(low=0, high=12500, size=(1,)).item()
        random_nums_used.append(rannum)

        img_cat = Image.open("kaggle_data/train/cat." + str(rannum) + ".jpg")
        img_dog = Image.open("kaggle_data/train/dog." + str(rannum) + ".jpg")
        test_cat = process_image(img_cat)
        test_dog = process_image(img_dog)
        test_loader.extend([(test_cat, [0]), (test_dog, [1])])

    # FINE-TUNE MODEL
    # CREATE THE MODEL
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    # Number of features for the last layer
    num_ftrs = model.fc.in_features
    # Assuming you want the output to be of 10 classes
    model.fc = nn.Linear(num_ftrs, 2)

    # Enable gradient computation for the newly created layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # TRAIN THE MODEL
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=5)

    # EVALUATE THE MODEL
    eval_model(trained_model, test_loader)


if __name__ == "__main__":
    main()
