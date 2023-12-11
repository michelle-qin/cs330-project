
import torch
from torch import optim
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
import random




class CustomResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50--try changing to resnet18 w only one linear layer
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Use the existing ResNet architecture up to the last layer
        x = self.resnet(x)
        return x


def train_model(model, train_loader, criterion, optimizer, writer, test_loader, num_epochs=1000):
    # If a GPU is available, move the model to GPU
    # probably the learning rate is too large
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over the training data
        random.shuffle(train_loader)
        optimizer.zero_grad()
        for inputs, labels in train_loader:
            inputs, labels = inputs.unsqueeze(dim=0).to(device), torch.tensor(
                labels
            ).to(device)
        
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)
        
        optimizer.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / len(train_loader)
        writer.add_scalar("training loss", epoch_loss, epoch)
        writer.add_scalar("training accuracy", epoch_acc, epoch)

        if epoch % 20 == 0:
            eval_acc, eval_loss = eval_model(model, criterion, test_loader)
            writer.add_scalar("evaluation loss", eval_loss, epoch)
            writer.add_scalar("evaluation accuracy", eval_acc, epoch)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f} Train_Acc: {epoch_acc:.4f}"
            )

    return model


def eval_model(model, criterion, test_loader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_corrects = 0
    total_samples = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.unsqueeze(dim=0).to(device), torch.tensor(labels).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs)
            running_corrects += torch.sum(preds == labels)
            total_samples += labels.size(0)

    total_acc = running_corrects.double() / total_samples
    eval_loss = running_loss / len(test_loader)
    print(f"Eval Loss: {eval_loss:.4f} Accuracy: {total_acc:.4f}")
    return (total_acc, eval_loss)


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

def retrieve_train_loader(train_dict, cache_folder):
    train_loader = []
    for animal in train_dict.keys():
        dir_name = cache_folder+'/' + animal + '/'
        image_files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
        if not image_files:
            print(f"No images found in the cache folder: {cache_folder}")
            return
        for image_file in image_files:
            image_path = os.path.join(dir_name, image_file)
            
            try:
                # Open the image
                image = Image.open(image_path)
                image_array = process_image(image)
                train_loader.extend(
                                [(image_array, train_dict[animal])]
                            )

            except Exception as e:
                # Handle any potential errors while opening images
                print(f"Error opening image {image_path}: {e}")
    print(len(train_loader))
    return train_loader
        


def main(args):
    # CREATE INITIAL DATA
    # TRAIN DATA
    # test 12500, dog 12499, cat 12499r
    log_dir = "five_class_runs/" + args.strategy
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_dict = {"cat": [0], "dog": [1], "chicken": [2], "spider": [3], "horse": [4]}
    
    extra_animals = ['chicken', 'spider', 'horse']

    


    test_loader = []
    for animal in extra_animals:
      print('retrieving', animal)
      animal_files = os.listdir(f"extra_animals/raw-img/{animal}/")
      selected_pics = random.sample(animal_files, 99)
      for img in selected_pics:
          pic = process_image(Image.open(f"extra_animals/raw-img/{animal}/{img}"))
          test_loader.extend([(pic, train_dict[animal])])

    # SUPPLEMENT DATA
    STRATEGY = args.strategy
    cache_folder = STRATEGY+"_cache"
    train_loader = retrieve_train_loader(train_dict, cache_folder)

            
    model = CustomResNet(num_classes=5)
    # TEST DATA
    random_nums_used = [4505, 5135]
    print(len(test_loader))
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


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.resnet.fc.parameters()), lr=0.0005)

    # TRAIN THE MODEL
    trained_model = train_model(
        model, train_loader, criterion, optimizer, writer, test_loader, num_epochs=1000
    )

    # EVALUATE THE MODEL
    total_acc, _ = eval_model(trained_model, criterion, test_loader)
    print(f'Final accuracy on test:{total_acc}')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    strategies = [
        "BASELINE",
        "RANDOM",
        "TEXT_RETRIEVAL",
        "SEMANTIC_NEAREST_NEIGHBOR",
        "CONTENT",
        "CONTENT_DIVERSE",
        "DIVERSE",
    ]
    parser.add_argument(
        "--strategy",
        type=str,
        default="BASELINE",
        choices=strategies,
        help="Choose a mode from the options: {}".format(", ".join(strategies)),
    )
    os.makedirs("five_class_runs", exist_ok=True)
    main(parser.parse_args())