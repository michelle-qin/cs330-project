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
import random

from transformers import BlipProcessor, BlipForConditionalGeneration
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)


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

horse_breeds = [
    "Arabian Horse",
    "Clydesdale",
    "Appaloosa",
    "Friesian",
    "Andalusian",
    "Shetland Pony",
    "Percheron",
    "Pinto Horse",
    "Mustang",
    "Haflinger",
    "Tennessee Walking Horse",
    "Morgan Horse",
    "Paint Horse",
    "Thoroughbred",
    "Akhal-Teke",
    "Gypsy Vanner",
    "Connemara Pony",
    "Hanoverian",
    "Welsh Pony",
    "Icelandic Horse"
]

chicken_breeds = [
    "Leghorn",
    "Orpington",
    "Silkie",
    "Easter Egger",
    "Wyandotte",
    "Serama",
    "Polish",
    "Rhode Island Red",
    "Australorp",
    "Frizzle",
    "Plymouth Rock",
    "Cochin",
    "Sebright",
    "Ayam Cemani",
    "Speckled Sussex",
    "Brahma",
    "Naked Neck",
    "Hamburg",
    "Barnevelder",
    "Faverolle"
]

spider_breeds = [
    "Golden Orb Weaver",
    "Jumping Spider",
    "Black Widow",
    "Tarantula",
    "Garden Orb Weaver",
    "Huntsman Spider",
    "Peacock Spider",
    "Trapdoor Spider",
    "Wolf Spider",
    "Orb-weaving Spider",
    "Daddy Longlegs",
    "Brown Recluse",
    "Crab Spider",
    "Net-Casting Spider",
    "Tarantula Hawk Spider Wasp",
    "Ladybird Spider",
    "Long-jawed Orb Weaver",
    "Redback Spider",
    "Bolas Spider",
    "Trapdoor Jumping Spider"
]

diverse_breeds = {"cat": cat_breeds, "dog": dog_breeds, "chicken": chicken_breeds, "spider": spider_breeds, "horse": horse_breeds}


class CustomResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50--try changing to resnet18 w only one linear layer
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
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


def eval_model(model, criterion, test_loader):
    model.eval()  # Set the model to evaluation mode

    running_corrects = 0
    total_samples = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.unsqueeze(dim=0), torch.tensor(labels)

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(image)  # (3, 224, 224)
    return img_tensor

def text_supplement_with_laion(unprocessed_train_loader, cache_folder, num_supplement=20):
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

        pet_folder = cache_folder + "/" + pet_name + "/"
        os.makedirs(pet_folder, exist_ok=True)

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
                        image_path = os.path.join(pet_folder, str(len(supplement_dict[pet_name])) + '.jpg')
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Image saved successfully at {image_path}")
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


def semantic_NN_supplement_with_laion(train_dict, train_images, cache_folder, num_supplement=20):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=1000,
    )

    supplement_dict = {}
    
    for pet_name in train_dict.keys():
        pet_folder = cache_folder + "/" + pet_name + "/"
        os.makedirs(pet_folder, exist_ok=True)
        embed = get_image_emb(train_images[pet_name])
        pet_images = client.query(embedding_input=embed.tolist())
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


def diverse_supplement_with_laion(train_dict, cache_folder, num_supplement=20):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
    )
    supplement_dict = {}
    for pet_name in train_dict.keys():
        pet_folder = cache_folder + "/" + pet_name + "/"
        os.makedirs(pet_folder, exist_ok=True)
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
                            image_path = os.path.join(pet_folder, str(len(supplement_dict[pet_name])) + '.jpg')
                            with open(image_path, 'wb') as f:
                                f.write(response.content)
                            print(f"Image saved successfully at {image_path}")
                        except Exception as e:
                            print("Issue with getting image: ", e)
                except requests.exceptions.RequestException as e:
                    print("FAIL")
                index += 1
    return supplement_dict


def content_supplement_with_laion(
    train_dict, source_data, source_paths, cache_folder, num_supplement=20, approach="closest"
):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=1000,
    )
    print(approach)
    supplement_dict = {}
    for pet_name in train_dict.keys():
        pet_folder = cache_folder + "/" + pet_name + "/"
        os.makedirs(pet_folder, exist_ok=True)
        pet_images = client.query(image=source_paths[pet_name])
        num_pet_images = len(pet_images)
        print('pet_name', num_pet_images)
        source_im = source_data[pet_name]
        intermediate_hund = []
        scores = []
        saved_content = []
        for i in range(30):
            # while len(supplement_dict[pet_name]) < num_supplement:
            image_path = pet_images[i]["url"]
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
                                net_type="squeeze",
                            )
                        )
                        saved_content.append(response.content)
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
        supplement_dict[pet_name] = []
        for item in final_tw:
            supplement_dict[pet_name].append((item, train_dict[pet_name]))
        for i in range(20):
            image_path = os.path.join(pet_folder, str(i) + '.jpg')
            with open(image_path, 'wb') as f:
                index = best_k[i]
                print(index)
                f.write(saved_content[index])
            print(f"Image saved successfully at {image_path}")

    return supplement_dict


def random_supplement_with_laion(train_dict, cache_folder, num_supplement=20):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        num_images=500,
    )
    supplement_dict = {}
    for pet_name in train_dict.keys():
        pet_folder = cache_folder + "/" + pet_name + "/"
        os.makedirs(pet_folder, exist_ok=True)
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
                        image_path = os.path.join(pet_folder, str(len(supplement_dict[pet_name])) + '.jpg')
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Image saved successfully at {image_path}")
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
    log_dir = "five_class_runs/" + args.strategy
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    cat_num = 4505
    dog_num = 5153
    cat_path = "kaggle_data/train/cat." + str(cat_num) + ".jpg"
    dog_path = "kaggle_data/train/dog." + str(dog_num) + ".jpg"
    print(cat_path)
    print(dog_path)
    
    extra_animals = ['chicken', 'spider', 'horse']

    img_cat = Image.open(cat_path)
    img_dog = Image.open(dog_path)
    train_cat = process_image(img_cat)
    train_dog = process_image(img_dog)

    train_data = [train_cat, train_dog]
    train_dict = {"cat": [0], "dog": [1], "chicken": [2], "spider": [3], "horse": [4]}
    train_labels = [[0], [1]]
    unprocessed_train_loader = [(img_cat, 0), (img_dog, 1)]
    train_loader = [(train_data[i], train_labels[i]) for i in range(len(train_data))]
    train_images = {"cat":img_cat, "dog": img_dog}

    chicken_path = f"extra_animals/raw-img/chicken/OIP-3C0nKePl9Pm-VqHKea5vYAAAAA.jpeg"
    chicken_pic = Image.open(f"extra_animals/raw-img/chicken/OIP-3C0nKePl9Pm-VqHKea5vYAAAAA.jpeg")
    train_chicken = process_image(chicken_pic)
    train_loader.extend([(process_image(chicken_pic), train_dict['chicken'])])
    train_images['chicken'] = chicken_pic

    spider_path = f"extra_animals/raw-img/spider/OIP-2nC_WOE8bzHIZSAYcvE8ZgHaFs.jpeg"
    spider_pic = Image.open(f"extra_animals/raw-img/spider/OIP-2nC_WOE8bzHIZSAYcvE8ZgHaFs.jpeg")
    train_spider = process_image(spider_pic)
    train_loader.extend([(process_image(spider_pic), train_dict['spider'])])
    train_images['spider'] = spider_pic

    horse_path = f"extra_animals/raw-img/horse/OIP-VpiOc6W4JGaIM39qYbpa0QHaFj.jpeg"
    horse_pic = Image.open(f"extra_animals/raw-img/horse/OIP-VpiOc6W4JGaIM39qYbpa0QHaFj.jpeg")
    train_horse = process_image(horse_pic)
    train_loader.extend([(process_image(horse_pic), train_dict['horse'])])
    train_images['horse'] = horse_pic


    test_loader = []
    for animal in extra_animals:
      print('retrieving', animal)
      animal_files = os.listdir(f"extra_animals/raw-img/{animal}/")
      selected_pics = random.sample(animal_files, 99)
      print(f"extra_animals/raw-img/{animal}/{selected_pics[0]}")
      for img in selected_pics:
          pic = process_image(Image.open(f"extra_animals/raw-img/{animal}/{img}"))
          test_loader.extend([(pic, train_dict[animal])])

    # SUPPLEMENT DATA
    STRATEGY = args.strategy
    cache_folder = STRATEGY+"_cache"
    print("USING STRATEGY: ", STRATEGY)
    if STRATEGY == "BASELINE":
        pass
    else:
        # supplement_data key: pet_name, value: list of (image, label) tuples
        if STRATEGY == "RANDOM":
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)
            supplement_data = random_supplement_with_laion(train_dict, cache_folder)
        elif STRATEGY == "TEXT_RETRIEVAL":
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)
                supplement_data = text_supplement_with_laion(unprocessed_train_loader, cache_folder=cache_folder)
            else:
                supplement_data = retrieve_cache_images(train_dict, cache_folder)
        elif STRATEGY == "SEMANTIC_NEAREST_NEIGHBOR":
            supplement_data = semantic_NN_supplement_with_laion(train_dict, train_images, cache_folder=cache_folder)
        elif STRATEGY == "DIVERSE":
            supplement_data = diverse_supplement_with_laion(train_dict, cache_folder)
        elif STRATEGY == "CONTENT":
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)
            source_data = {"cat": train_data[0], "dog": train_data[1], "chicken": train_chicken, "spider": train_spider, "horse": train_horse}
            source_paths = {"cat": cat_path, "dog": dog_path, "chicken": chicken_path, "spider": spider_path, "horse": horse_path}
            supplement_data = content_supplement_with_laion(train_dict, source_data, source_paths, cache_folder)
        elif STRATEGY == "CONTENT_DIVERSE":
            source_data = {"cat": train_data[0], "dog": train_data[1]}
            source_paths = {"cat": cat_path, "dog": dog_path}
            supplement_data = content_supplement_with_laion(train_dict, source_data, source_paths, approach="furthest")
        for pet_name in train_dict.keys():
            train_loader.extend(supplement_data[pet_name])
    


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