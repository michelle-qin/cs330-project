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
import os
import numpy as np
import argparse
import random
from torch.utils.tensorboard import SummaryWriter

from transformers import BlipProcessor, BlipForConditionalGeneration
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)

# STRATEGY = "RANDOM"
# STRATEGY = "BASELINE"
# STRATEGY = "TEXT_RETRIEVAL"
# STRATEGY = "SEMANTIC_NEAREST_NEIGHBOR"
# STRATEGY = "DIVERSE_IMAGES"


cat_breeds = [
    "Persian Cat",
    "Siamese Cat",
    "Maine Coon",
    "Sphynx Cat",
    "Bengal Cat",
    "Scottish Fold",
    "Ragdoll Cat",
    "Burmese Cat",
    "Abyssinian Cat",
    "Russian Blue",
    "Cornish Rex",
    "Norwegian Forest Cat",
    "Manx Cat",
    "Egyptian Mau",
    "Munchkin Cat",
    "Turkish Van",
    "Himalayan Cat",
    "Chartreux Cat",
    "Oriental Shorthair",
    "American Shorthair",
]

dog_breeds = [
    "Labrador Retriever",
    "German Shepherd",
    "Golden Retriever",
    "Bulldog",
    "Poodle",
    "Beagle",
    "Boxer",
    "Dachshund",
    "Siberian Husky",
    "Great Dane",
    "Shih Tzu",
    "Chihuahua",
    "Rottweiler",
    "Doberman Pinscher",
    "Pomeranian",
    "Border Collie",
    "King Charles Spaniel",
    "Australian Shepherd",
    "Pug",
    "Dalmatian",
]

bird_descriptions = [
    "Majestic eagle soars above mountains, piercing eyes and powerful beak.",
    "Tiny hummingbird hovers near red flower, iridescent feathers shimmering.",
    "Playful pigeons strut across city street, gray feathers blending with concrete.",
    "Bright yellow canary sings joyfully in its cage, tiny body vibrating with music.",
    "Blue heron patiently stands in shallow pond, long legs reflected in the water.",
    "Pink flamingos dance across lake, long legs and necks creating a spectacle.",
    "Shy owl peeks from tree hollow, golden eyes and white feathers camouflaged.",
    "Chattering starlings form a swirling black cloud in the sky, movements synchronized.",
    "Proud peacock struts across lawn, iridescent plumage spreading like a fan.",
    "Red-breasted robins chirp in their nest, surrounded by green leaves.",
    "Snow geese paint the sky white as they migrate south, honking calls echoing.",
    "Lone penguin waddles across snowy landscape, black and white contrasting with white.",
    "Loving pair of birds perch on a branch, heads nuzzling, feathers vibrant.",
    "Tiny hummingbird moth hovers over lavender, wings mimicking the bird's movements.",
    "Colorful parrots squawk in rainforest canopy, feathers adding splash of color.",
    "Majestic condor circles above mountains, massive wings spanning ten feet, shadow falling.",
    "Sandpiper searches for food along seashore, long legs wading through shallow water.",
    "Elegant swans glide serenely across lake, white feathers reflecting peaceful tranquility.",
    "Crows gather in barren field, black feathers stark against brown earth, cawing calls echoing.",
    "Tiny wren builds its intricate nest inside a hanging basket, small body flitting with twigs.",
]


drone_descriptions = [
    "Sleek black drone zips through the city skyline, capturing breathtaking aerial views.",
    "Vibrant yellow drone hovers above a field of sunflowers, pollinating the flowers as it flies.",
    "Silent white drone glides effortlessly through the snowy mountains, its wings barely audible.",
    "Powerful red drone delivers medical supplies to a remote village, its mission critical.",
    "Agile blue drone dances through the air like a butterfly, its movements mesmerizing.",
    "Sturdy green drone surveys the aftermath of a natural disaster, providing vital information.",
    "Stealthy black drone patrols the border at night, its sensors scanning for intruders.",
    "Playful orange drone chases a group of children through the park, filling the air with laughter.",
    "Capable grey drone assists firefighters in battling a blaze, its heat sensors guiding the way.",
    "Curious purple drone explores a vibrant coral reef, discovering new species of marine life.",
    "High-tech silver drone conducts scientific research in the Arctic, braving harsh conditions.",
    "Industrious brown drone delivers packages to homes across the city, its efficiency unmatched.",
    "Elegant gold drone performs a breathtaking aerial ballet, its movements synchronized to music.",
    "Tiny pink drone carries a precious seed across a vast desert, planting new life in barren lands.",
    "Nimble blue drone flies through a dense forest canopy, mapping the terrain with pinpoint accuracy.",
    "Courageous green drone searches for survivors in the aftermath of a disaster, its tireless efforts saving lives.",
    "Eerie grey drone hovers silently above a battlefield, its presence a symbol of destruction.",
    "Hopeful white drone drops food and water to refugees in a war-torn country, its mission humanitarian.",
    "Futuristic black drone delivers a package directly to your doorstep, revolutionizing delivery services.",
    "Innovative silver drone explores the depths of a volcanic crater, revealing the secrets of the Earth.",
]

diverse_breeds = {"bird": bird_descriptions, "drone": drone_descriptions}


class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50--try changing to resnet18 w only one linear layer
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)
        print("resnet", self.resnet)

        # Remove the last fully connected layer of the ResNet model
        # self.resnet.fc = nn.Identity()
        # Freeze all layers
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # don't do this inside the class!
        # Replace the last fully connected layer
        # self.fc1 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Use the existing ResNet architecture up to the last layer
        x = self.resnet(x)
        # print(x.shape)
        # Apply the two new fully connected layers
        # x = self.fc1(x)
        # x = nn.ReLU()(x)  # You need a non-linear activation function here
        # x = self.fc2(x)
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
            # Move inputs and labels to the same device as the model
            # batch size for resnet is a lot better and needs to be a really small learning rate
            # should shuffle the cat and dogs!! and shuffle at every epoch!
            inputs, labels = inputs.unsqueeze(dim=0).to(device), torch.tensor(
                labels
            ).to(device)

            # Zero the parameter gradients
            

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # print(preds, labels)
            loss = criterion(outputs, labels)
            # print(loss)

            # Backward pass and optimize
            loss.backward()

            # Statistics
            running_loss += loss.item()
            # * inputs.size(0)
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


def eval_model(model, criterion, test_loader):
    model.eval()  # Set the model to evaluation mode

    running_corrects = 0
    total_samples = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.unsqueeze(dim=0), torch.tensor(labels)
            # print("INPUTS: ", inputs)
            # print("LABELS: ", labels)

            outputs = model(inputs)
            # print("OUTPUTS: ", outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs)
            # print("PREDS: ", preds)
            # print("LABELS: ", labels)
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
        ]
    )
    img_tensor = preprocess(image)  # (3, 224, 224)
    if img_tensor.shape[0] == 1:
        img_tensor = torch.cat([img_tensor, img_tensor, img_tensor])
    img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.226])(img_tensor)
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
            pet_name = "bird"
        else:
            pet_name = "drone"
        caption = generate_caption(img)

        pet_images = client.query(text=caption)
        num_pet_images = len(pet_images)

        # pet_folder = cache_folder + "/" + pet_name + "/"
        # os.makedirs(pet_folder, exist_ok=True)

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
                        # image_path = os.path.join(pet_folder, str(len(supplement_dict[pet_name])) + '.jpg')
                        # with open(image_path, 'wb') as f:
                        #     f.write(response.content)
                        # print(f"Image saved successfully at {image_path}")
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


def semantic_NN_supplement_with_laion(train_dict, img_bird, img_drone, cache_folder, num_supplement=20):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=4000,
    )
    bird_embed = get_image_emb(img_bird)
    drone_embed = get_image_emb(img_drone)

    supplement_dict = {}
    
    for pet_name in train_dict.keys():
        pet_folder = cache_folder + "/" + pet_name + "/"
        os.makedirs(pet_folder, exist_ok=True)
        print(bird_embed)
        print(bird_embed.shape)
        if pet_name == "bird":
            pet_images = client.query(embedding_input=bird_embed.tolist())
        else:
            pet_images = client.query(embedding_input=drone_embed.tolist())
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
                        image_path = os.path.join(pet_folder, str(len(supplement_dict[pet_name])) + '.jpg')
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Image saved successfully at {image_path}")
                    except Exception as e:
                        print("Issue with getting image: ", e)
            except requests.exceptions.RequestException as e:
                print("FAIL")
            i += 1

    return supplement_dict


def diverse_supplement_with_laion(train_dict, num_supplement=20):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
    )
    supplement_dict = {}
    for pet_name in train_dict.keys():
        diverse_names = diverse_breeds[pet_name]
        supplement_dict[pet_name] = []
        for i in range(len(diverse_names)):
            pet_image = client.query(text="an image of a " + diverse_names[i])
            print("hello", diverse_names[i])
            num_pet_images = len(pet_image)
            print(num_pet_images)
            index = 0
            accepted = False

            while not accepted:
                image_path = pet_image[index]["url"]
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
                            accepted = True
                        except Exception as e:
                            print("Issue with getting image: ", e)
                except requests.exceptions.RequestException as e:
                    print("FAIL")
                index += 1
    return supplement_dict


def content_supplement_with_laion(
    train_dict, source_data, source_paths, num_supplement=20, approach="closest"
):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=4000,
    )
    supplement_dict = {}
    for pet_name in train_dict.keys():
        pet_images = client.query(text="an image of a "+ pet_name)
        num_pet_images = len(pet_images)
        print(num_pet_images)
        source_im = source_data[pet_name]
        intermediate_hund = []
        scores = []
        i = 0
        while len(intermediate_hund) < 100:
            image_path = pet_images[i]["url"]
            i += 1
            print("IMAGE PATH: ", image_path)
            try:
                response = requests.get(image_path)
                if response.status_code == 200:
                    try:
                        image = Image.open(io.BytesIO(response.content))
                        image_array = process_image(image)
                        intermediate_hund.append(image_array.unsqueeze(dim=0))
                        max_abs = image_array.max()
                        if image_array.min() < 0:
                            max_abs = max(max_abs, -image_array.min())
                        image_array = image_array / max_abs
                        max_abs = source_im.max()
                        if source_im.min() < 0:
                            max_abs = max(max_abs, -source_im.min())
                        source_im = source_im / max_abs
                        scores.append(
                            learned_perceptual_image_patch_similarity(
                                source_im.unsqueeze(dim=0),
                                image_array.unsqueeze(dim=0),
                                net_type="alex",
                            )
                        )
                    except Exception as e:
                        print("Issue with getting image: ", e)
            except requests.exceptions.RequestException as e:
                print("FAIL")

        scores = torch.tensor(scores)
        if approach == "furthest":
            best_k = torch.argsort(scores)[(100-num_supplement):]
        elif approach == "closest":
            best_k = torch.argsort(scores)[:num_supplement]
        intermediate_hund = torch.cat(intermediate_hund)
        final_tw = intermediate_hund[best_k]
        supplement_dict[pet_name] = []
        for item in final_tw:
            supplement_dict[pet_name].append((item, train_dict[pet_name]))

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

def retrieve_cache_images(train_dict, cache_folder):
    supplement_dict = {}
    for pet_name in train_dict.keys():
        dir_name = cache_folder+'/' + pet_name + '/'
        supplement_dict[pet_name] = []
        image_files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
        print(image_files)
        if not image_files:
            print(f"No images found in the cache folder: {cache_folder}")
            return

        # Iterate through the image files and open them
    
        for image_file in image_files:
            image_path = os.path.join(dir_name, image_file)
            
            try:
                # Open the image
                image = Image.open(image_path)
                image_array = process_image(image)
                supplement_dict[pet_name].append(
                                (image_array, train_dict[pet_name])
                            )

            except Exception as e:
                # Handle any potential errors while opening images
                print(f"Error opening image {image_path}: {e}")
    return supplement_dict


def main(args):
    # CREATE INITIAL DATA
    # TRAIN DATA
    # test 12500, dog 12499, cat 12499r
    log_dir = "runs/birds-drones/" + args.strategy
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    image_folder = "BirdVsDrone/Drones"
    # Get all image paths in the folder
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
    # Choose a random image path
    random_image_path = random.choice(image_paths)
                                  
    bird_num = torch.randint(low=0, high=400, size=(1,)).item()
    bird_path = "BirdVsDrone/Birds/singleBirdinsky" + str(bird_num) + ".jpeg"
    drone_path = random_image_path

    img_bird = Image.open(bird_path)
    img_drone = Image.open(drone_path)
    train_bird = process_image(img_bird)
    train_drone = process_image(img_drone)

    train_data = [train_bird, train_drone]
    train_dict = {"bird": [0], "drone": [1]}
    train_labels = [[0], [1]]
    unprocessed_train_loader = [(img_bird, 0), (img_drone, 1)]
    train_loader = [(train_data[i], train_labels[i]) for i in range(len(train_data))]
    
    # SUPPLEMENT DATA
    STRATEGY = args.strategy
    cache_folder = STRATEGY+"_cache"
    print("USING STRATEGY: ", STRATEGY)
    if STRATEGY == "BASELINE":
        pass
    else:
        # supplement_data key: pet_name, value: list of (image, label) tuples
        if STRATEGY == "RANDOM":
            # if not os.path.exists(cache_folder):
            #     os.makedirs(cache_folder)
            #     supplement_data = random_supplement_with_laion(train_dict, cache_folder)
            # else:
            #     supplement_data = retrieve_cache_images(train_dict, cache_folder)
            supplement_data = random_supplement_with_laion(train_dict)
        elif STRATEGY == "TEXT_RETRIEVAL":
            # if not os.path.exists(cache_folder):
            #     os.makedirs(cache_folder)
            #     supplement_data = text_supplement_with_laion(unprocessed_train_loader, cache_folder=cache_folder)
            # else:
            #     supplement_data = retrieve_cache_images(train_dict, cache_folder)
            supplement_data = text_supplement_with_laion(unprocessed_train_loader)
        elif STRATEGY == "SEMANTIC_NEAREST_NEIGHBOR":
            # KATE TO DO
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)
                supplement_data = semantic_NN_supplement_with_laion(train_dict, img_bird, img_drone, cache_folder=cache_folder)
            else:
                supplement_data = retrieve_cache_images(train_dict, cache_folder)
        elif STRATEGY == "DIVERSE":
            # KATE TO DO
            supplement_data = diverse_supplement_with_laion(train_dict)
        elif STRATEGY == "CONTENT":
            source_data = {"bird": train_data[0], "drone": train_data[1]}
            source_paths = {"bird": bird_path, "drone": drone_path}
            supplement_data = content_supplement_with_laion(train_dict, source_data, source_paths)
        elif STRATEGY == "CONTENT_DIVERSE":
            source_data = {"bird": train_data[0], "drone": train_data[1]}
            source_paths = {"bird": bird_path, "drone": drone_path}
            supplement_data = content_supplement_with_laion(train_dict, source_data, source_paths, approach="furthest")
        for pet_name in train_dict.keys():
            train_loader.extend(supplement_data[pet_name])

            
    model = CustomResNet(num_classes=2)
    # TEST DATA
    random_nums_used = [bird_num]
    drone_image_used = [random_image_path]
    test_loader = []
    for i in range(99):
        rannum = torch.randint(low=0, high=400, size=(1,)).item()
        while rannum in random_nums_used:
            rannum = torch.randint(low=1, high=291, size=(1,)).item()
        random_nums_used.append(rannum)
        img_bird = Image.open("BirdVsDrone/Birds/singleBirdinsky" + str(rannum) + ".jpeg")

        random_image_path = random.choice(image_paths)
        while random_image_path in drone_image_used:
            random_image_path = random.choice(image_paths)
        drone_image_used.append(random_image_path)
        img_drone = Image.open(random_image_path)
        test_bird = process_image(img_bird)
        test_drone = process_image(img_drone)
        test_loader.extend([(test_bird, [0]), (test_drone, [1])])

    # FINE-TUNE MODEL
    # CREATE THE MODEL

    # Enable gradient computation for the newly created layers
    # for param in model.fc1.parameters():
    #     param.requires_grad = True

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(
    #     list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=0.0000001
    # )
    # model.resnet.fc.parameters()
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
    os.makedirs("runs", exist_ok=True)
    main(parser.parse_args())
