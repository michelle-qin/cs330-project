from mann import MANN
from clip_retrieval.clip_client import ClipClient, Modality
import numpy as np
import requests
import numpy as np
from PIL import Image
import io 
import torch
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import time
from transformers import BlipProcessor, BlipForConditionalGeneration

client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14")
all_pet_categories = [
"dog", # 7 images
"cat", # 23 images
"fish",
"bird", 
"snake",
"hamster",
"frog",
"rabbit",
"horse",
"turtle",
"lizard",
"mouse",
"pig",
"skunk",
"chinchilla",
"gorilla",
"giraffe",
"monkey",
"zebra",
"penguin",
"bear",
"deer",
"ant",
]
train_pet_categories = [
"dog", # 7 images
"cat", # 23 images
]
test_pet_categories = [
    "bear",
    "zebra"
]
num_shot = 2
num_way = 2
learning_rate = 1e-3
meta_train_steps = 2
meta_batch_size = 1
random_seed = 123
hidden_dim = 128

# Image Captioning Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def label_to_one_hot(label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

def get_images(pet_categories):
    image_and_labels = []

    # Create support set 
    for label_idx in range(num_way):
        pet = pet_categories[label_idx]
        pet_images = client.query(text="an image of a " + pet)
        samples = []

        i = 0 
        while len(samples) < num_shot + 1:
            image_path = pet_images[i]['url']
            response = requests.get(image_path)
            if response.status_code == 200:
                try:
                    image = Image.open(io.BytesIO(response.content))

                    image_array = np.asarray(image)
                    image_array = image_file_to_array(image_array)

                    one_hot_label = label_to_one_hot(label_idx, num_way)
                    samples.append((one_hot_label, image_array))
                except: 
                    print("Issue with getting image")
            i += 1

        image_and_labels.extend(samples)
    
    return image_and_labels

def image_file_to_array(image_array):
    """
    Takes an image array and returns numpy array.
    Args:
        image_array: Image array (numpy array)
    Returns:
        Flattened 1D image array
    """
    # Calculate the expected width and height (assuming a square image)
    # expected_size = int((dim_input/3) ** 0.5)  # Calculate expected size (e.g., 28 for 784)
    # import pdb
    # pdb.set_trace()
    expected_size = 256
    # Resize the image if it's not already the expected size
    # if image_array.shape[0] != expected_size or image_array.shape[1] != expected_size:
    image = Image.fromarray(image_array)
    image = image.resize((expected_size, expected_size))
    image_array = np.asarray(image)

    # Flatten and normalize the image
    image_array = image_array.astype(np.float32) / 255.0  # Normalize the array
    image_array = 1.0 - image_array
    image_array = image_array.reshape(-1)  # Flatten the array
    return image_array

def _sample(labels_and_images):
    k_images = [[] for _ in range(num_shot + 1)]
    k_labels = [[] for _ in range(num_shot + 1)]
    shot_num = 0

    for label_and_image in labels_and_images:
        label = label_and_image[0] 
        image = label_and_image[1] 
        k_images[shot_num].append(image)
        k_labels[shot_num].append(label)
        shot_num += 1 
        if shot_num == num_shot + 1: 
            shot_num = 0
   
    for i in range(num_shot):
        k_images[i] = np.asarray(k_images[i])
        k_labels[i] = np.asarray(k_labels[i])

    image_batch = np.asarray(k_images)
    print("IMAGE BATCH: ", image_batch)
    label_batch = np.asarray(k_labels)
    
    # Step 4: Shuffle the order of examples from the query set
    num_items_query_set = np.arange(image_batch[-1].shape[0])
    np.random.shuffle(num_items_query_set)
    query_images = image_batch[-1][num_items_query_set]
    query_labels = label_batch[-1][num_items_query_set]
    image_batch[-1] = query_images
    label_batch[-1] = query_labels

    # Step 5: return tuple of image batch with shape [K+1, N, 784] and
    #         label batch with shape [K+1, N, N]
    return (image_batch, label_batch)

def meta_train_step(images, labels, model, optim, eval=False):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()

if __name__ == "__main__":

    writer = SummaryWriter(f"runs/{num_way}_{num_shot}_{random_seed}_{hidden_dim}")
    
    train_labels_and_images = get_images(train_pet_categories)
    test_labels_and_images = get_images(test_pet_categories)
    train_image_batch, train_label_batch = _sample(train_labels_and_images)
    test_image_batch, test_label_batch = _sample(test_labels_and_images)

    train_image_batch = torch.from_numpy(train_image_batch)
    train_label_batch = torch.from_numpy(train_label_batch)
    K, N, train_image_size = train_image_batch.size()
    train_image_batch = train_image_batch.reshape((1, K, N, train_image_size))
    train_label_batch = train_label_batch.reshape((1, K, N, N))

    print("TRAIN IMAGE BATCH: ", train_image_batch) #TODO I notice there is a discrepancy between IMAGE BATCH and TRAIN IMAGE BATCH?
    # SUPPLEMENT THE DATA HERE
    supplemented_data = []
    for batch in train_image_batch:
        print("BATCH: ", batch)
        for image_tensor in batch:
            print("IMAGE TENSOR: ", image_tensor)
            # convert the tensor to PIL Image for caption generation
            image = Image.fromarray(image_tensor.numpy().astype(np.uint8))
            # generate caption
            caption = generate_caption(image)
            print("CAPTION: ", caption)
            # store the image tensor and its caption
            supplemented_data.append((image_tensor, caption))
    print(supplemented_data)
    
    quit()
    
    test_image_batch = torch.from_numpy(test_image_batch)
    test_label_batch = torch.from_numpy(test_label_batch)
    K, N, test_image_size = test_image_batch.size()
    test_image_batch = test_image_batch.reshape((1, K, N, test_image_size))
    test_label_batch = test_label_batch.reshape((1, K, N, N))

    model = MANN(num_way, num_shot + 1, hidden_dim)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    times = []
    
    for step in tqdm(range(meta_train_steps)):
        # Sample batch
        t0 = time.time()
        i, l = train_image_batch, train_label_batch
        t1 = time.time()
        # Train
        _, ls = meta_train_step(i, l, model, optim)
        t2 = time.time()
        print("Loss/train", ls, step)
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])

        # Evaluate 
        i, l = test_image_batch, test_label_batch
        pred, tls = meta_train_step(i, l, model, optim, eval=True)   
        print("Train Loss:", ls.cpu().numpy(), "Test Loss:", tls.cpu().numpy())     
        print("Loss/test", tls, step)
        writer.add_scalar("Loss/test", tls, step)
        pred = torch.reshape(
            pred, [-1, num_shot + 1, num_way, num_way]
        )

        pred = torch.argmax(pred[:, -1, :, :], axis=2)
        l = torch.argmax(l[:, -1, :, :], axis=2)
        acc = pred.eq(l).sum().item() / (meta_batch_size * num_way)
        print("Test Accuracy", acc)
        print("Accuracy/test", acc, step)
        writer.add_scalar("Accuracy/test", acc, step)

        times = np.array(times)
        print(f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}")
        times = []


