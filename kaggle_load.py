import matplotlib.pyplot as plt
import torch
from torch import optim
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms 
    
def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
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
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def eval_model(model, test_loader):
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader: 
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    total_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Eval Accuracy: {total_acc}')

def process_image(image_path):
    # Load and preprocess an image 
    img = Image.open(image_path) 
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    img_tensor = preprocess(img) 
    return img_tensor

def main():
    # CREATE INITIAL DATA 
    # TRAIN DATA
    #test 12500, dog 12499, cat 12499r
    cat_num = torch.randint(low=0, high=12499, size=(1,)).item()
    dog_num = torch.randint(low=0, high=12499, size=(1,)).item()

    train_cat = process_image("kaggle_data/train/cat." + str(cat_num) + ".jpg")
    train_dog = process_image("kaggle_data/train/dog." + str(dog_num) + ".jpg")
    train_data = [train_cat, train_dog]
    train_labels = [[0, 1], [1, 0]]
    train_loader = [(train_data[i], train_labels[i]) for i in range(len(train_data))]

    # TEST DATA
    random_nums_used = []
    test_loader = []
    for i in range(99):
        rannum = torch.randint(low=0, high=12500, size=(1,)).item()
        while rannum in random_nums_used:
            rannum = torch.randint(low=0, high=12500, size=(1,)).item()
        random_nums_used.append(rannum)
        test_cat = process_image("kaggle_data/test/cat." + str(rannum) + ".jpg")
        test_dog = process_image("kaggle_data/test/dog." + str(rannum) + ".jpg")
        test_loader.extend([(test_cat, [0, 1]), (test_dog, [1, 0])]) 

    # FINE-TUNE MODEL 
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
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=25)

    # EVALUATE THE MODEL
    eval_model(trained_model, test_loader)


if __name__ == "__main__":
    main()