import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to generate embeddings
def generate_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(image)
        
    return embedding.squeeze().numpy()

# Directory containing all class folders
data_dir = './data/'

# Dictionary to store embeddings by class
embeddings = {}

# Loop through each class folder (e.g., COVID, Normal, Lung_Opacity)
for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    
    # Make sure it's a directory
    if os.path.isdir(class_path):
        embeddings[class_folder] = []
        
        # Loop through each image in the class folder
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            # Check if it's an image file (you can add more formats if needed)
            if img_file.endswith(".png") or img_file.endswith(".jpg"):
                embedding = generate_embedding(img_path)
                embeddings[class_folder].append(embedding)
                print(f"Generated embedding for {img_file} in {class_folder}, shape: {embedding.shape}")

# Save the embeddings to a pickle file
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print("Embeddings generated and saved to embeddings.pkl")
