import matplotlib.pyplot as plt
import torch
from torch import optim
from PIL import Image
import torch.nn as nn
import clip
from torchvision import models, transforms
import pandas as pd
from clip_retrieval.clip_client import ClipClient
import requests
from PIL import Image
import io
import numpy as np

from transformers import BlipProcessor, BlipForConditionalGeneration
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)

STRATEGY = "RANDOM"
# STRATEGY = "BASELINE"
# STRATEGY = "TEXT_RETRIEVAL"
# STRATEGY = "SEMANTIC_NEAREST_NEIGHBOR"


class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        # Remove the last fully connected layer of the ResNet model
        self.resnet.fc = nn.Identity()
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last fully connected layer
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Use the existing ResNet architecture up to the last layer
        x = self.resnet(x)
        # Apply the two new fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)  # You need a non-linear activation function here
        x = self.fc2(x)
        return x


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


# def eval_model(model, test_loader):
#     running_corrects = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.unsqueeze(dim=0), torch.tensor(labels)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             print(preds, labels)
#             print(preds == labels)
#             running_corrects += torch.sum(preds == labels)
#     total_acc = running_corrects.double() / len(test_loader)
#     print(f"Eval Accuracy: {total_acc}")


def eval_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode

    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.unsqueeze(dim=0), torch.tensor(labels)
            print("INPUTS: ", inputs)
            print("LABELS: ", labels)

            outputs = model(inputs)
            print("OUTPUTS: ", outputs)
            _, preds = torch.max(outputs, 1)
            print("PREDS: ", preds)
            print("LABELS: ", labels)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    total_acc = running_corrects.double() / total_samples
    print(f"Eval Accuracy: {total_acc:.4f}")


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


def text_supplement_with_laion(unprocessed_train_loader, num_supplement=20):
    # MICHELLE
    # TO DO TRY IMAGES WITHOUT PROCESSING
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=500,
    )
    supplement_dict = {}
    # supplement_data key: pet_name, value: list of (image, label) tuples

    # Image Captioning Model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    def generate_caption(image):
        inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        print("CAPTION: ", caption)
        return caption

    for sample in unprocessed_train_loader:
        img = sample[0]
        label = sample[1]
        if label == 0:
            pet_name = "cat"
        else:
            pet_name = "dog"
        caption = generate_caption(img)

        pet_images = client.query(text=caption)
        num_pet_images = len(pet_images)

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
                        supplement_dict[pet_name].append((image_array, [label]))
                    except Exception as e:
                        print("Issue with getting image: ", e)
            except requests.exceptions.RequestException as e:
                print("FAIL")
    print(supplement_dict)
    return supplement_dict


def get_image_emb(image):
    model, preprocess = clip.load("ViT-L/14", device="cpu", jit=True)
    with torch.no_grad():
        image_emb = model.encode_image(preprocess(image).unsqueeze(0).to("cpu"))
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
        return image_emb


def semantic_NN_supplement_with_laion(train_dict, img_cat, img_dog, num_supplement=20):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=500,
    )
    cat_embed = get_image_emb(img_cat)
    dog_embed = get_image_emb(img_dog)

    supplement_dict = {}
    for pet_name in train_dict.keys():
        print(cat_embed)
        print(cat_embed.shape)
        if pet_name == "cat":
            pet_images = client.query(embedding_input=cat_embed.tolist())
        else:
            pet_images = client.query(embedding_input=dog_embed.tolist())
        num_pet_images = len(pet_images)
        print("num_pet_images", num_pet_images)
        supplement_dict[pet_name] = []
        i = 0
        while len(supplement_dict[pet_name]) < num_supplement:
            image_path = pet_images[i]["url"]
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
                        print(len(supplement_dict[pet_name]))
                    except Exception as e:
                        print("Issue with getting image: ", e)
            except requests.exceptions.RequestException as e:
                print("FAIL")
            i += 1

    return supplement_dict


def content_supplement_with_laion(
    train_dict, source_data, num_supplement=20, approach="closest"
):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=500,
    )
    supplement_dict = {}
    for pet_name in train_dict.keys():
        pet_images = client.query(text="an image of a " + pet_name)
        num_pet_images = len(pet_images)
        source_im = source_data[pet_name]
        intermediate_hund = []
        scores = []
        for i in range(100):
            # while len(supplement_dict[pet_name]) < num_supplement:
            image_path = pet_images[i]["url"]
            print("IMAGE PATH: ", image_path)
            try:
                response = requests.get(image_path)
                if response.status_code == 200:
                    try:
                        image = Image.open(io.BytesIO(response.content))
                        image_array = process_image(image)
                        min_im_ar = image_array.min()
                        max_im_ar = image_array.max()
                        normalized_tensor = torch.nn.functional.normalize(
                            image_array - min_im_ar, p=float("inf"), dim=0
                        )
                        image_array = normalized_tensor * (max_im_ar - min_im_ar)
                        intermediate_hund.append((image_array, train_dict[pet_name]))
                        scores.append(
                            learned_perceptual_image_patch_similarity(
                                source_im, image_array, net_type="squeeze"
                            )
                        )
                    except Exception as e:
                        print("Issue with getting image: ", e)
            except requests.exceptions.RequestException as e:
                print("FAIL")

        scores = torch.tensor(scores)
        if approach == "furthest":
            best_k = torch.topk(scores, num_supplement)
        elif approach == "closest":
            best_k = torch.argsort(scores)[:num_supplement]
        intermediate_hund = torch.cat(intermediate_hund)
        final_tw = intermediate_hund[best_k]

        supplement_dict[pet_name] = final_tw

    return supplement_dict


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
                    except Exception as e:
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
    unprocessed_train_loader = [(img_cat, 0), (img_dog, 1)]
    train_loader = [(train_data[i], train_labels[i]) for i in range(len(train_data))]

    # SUPPLEMENT DATA
    print("USING STRATEGY: ", STRATEGY)
    if STRATEGY == "BASELINE":
        pass
    else:
        # supplement_data key: pet_name, value: list of (image, label) tuples
        if STRATEGY == "RANDOM":
            supplement_data = random_supplement_with_laion(train_dict)
        elif STRATEGY == "TEXT_RETRIEVAL":
            supplement_data = text_supplement_with_laion(unprocessed_train_loader)
        elif STRATEGY == "SEMANTIC_NEAREST_NEIGHBOR":
            # KATE TO DO
            supplement_data = semantic_NN_supplement_with_laion(
                train_dict, img_cat, img_dog
            )
        elif STRATEGY == "CONTENT":
            source_data = {"cat": train_data[0], "dog": train_data[1]}
            supplement_data = content_supplement_with_laion(train_dict, source_data)
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
    model = CustomResNet(num_classes=2)

    # Enable gradient computation for the newly created layers
    for param in model.fc1.parameters():
        param.requires_grad = True
    for param in model.fc2.parameters():
        param.requires_grad = True

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=0.001
    )

    # TRAIN THE MODEL
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=5)

    # EVALUATE THE MODEL
    eval_model(trained_model, test_loader)


if __name__ == "__main__":
    main()
