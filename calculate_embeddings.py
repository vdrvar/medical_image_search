import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
import pickle
import json
import csv

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode since we are not training it

# Define the image transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

# Function to generate embedding for a single image
def generate_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    
    with torch.no_grad():
        embedding = model(image)
        
    return embedding.squeeze().numpy()  # Remove extra dimensions and convert to numpy array

# Function to generate and save embeddings for a specific class directory
def generate_class_embeddings(class_folder, output_dir):
    class_path = os.path.join(data_dir, class_folder)
    embeddings = []  # List to store embeddings for all images in the class
    metadata = []  # List to store metadata for all images in the class
    error_log_path = os.path.join(output_dir, 'error_log.txt')  # Path to the error log file
    
    # Loop through each file in the class directory
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        
        # Check if it's an image file (you can add more formats if needed)
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            try:
                embedding = generate_embedding(img_path)
                embeddings.append(embedding)  # Append the embedding to the list
                metadata.append({"filename": img_file, "shape": embedding.shape})
                print(f"Generated embedding for {img_file} in {class_folder}, shape: {embedding.shape}")
            except (UnidentifiedImageError, IOError) as e:
                # Log the error if the image cannot be processed
                with open(error_log_path, 'a') as log_file:
                    log_file.write(f"Error processing file {img_file} in {class_folder}: {str(e)}\n")
                print(f"Error processing file {img_file} in {class_folder}, logged to {error_log_path}")

    # Save the embeddings to a CSV file for each class
    output_file = os.path.join(output_dir, f'{class_folder}_embeddings.csv')
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for embedding in embeddings:
            csv_writer.writerow(embedding)
    print(f"Embeddings for class '{class_folder}' saved to {output_file}")

    # Save the metadata to a JSON file
    metadata_file = os.path.join(output_dir, f'{class_folder}_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata for class '{class_folder}' saved to {metadata_file}")

# Directory containing all class folders
data_dir = './data/'
output_dir = './embeddings/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each class folder and generate embeddings
for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(class_path):
        generate_class_embeddings(class_folder, output_dir)

print("Embeddings generated and saved to separate files.")
