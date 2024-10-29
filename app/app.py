import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import os
from annoy import AnnoyIndex
from collections import Counter

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode

# Define transformations (same as used in training embeddings)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class name to prettier label mapping
class_labels = {
    "COVID": "COVID-19",
    "Lung": "Lung Opacity",
    "Normal": "Normal",
    "Viral": "Viral Pneumonia"
}

# Display supported classes
supported_classes = ", ".join(class_labels.values())

# Streamlit App Title and Description
st.title("Medical Image Classification with Annoy (Approximate Nearest Neighbors)")
st.write("Upload an X-ray image, set the number of neighbors (k), and get a classification.")
st.markdown("### Supported Classes:")
st.markdown("""
- **COVID-19**
- **Lung Opacity**
- **Normal**
- **Viral Pneumonia**
""")

# Function to generate embedding for a single image
def generate_embedding(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze().numpy()

# Load embeddings and labels from pickle files
embeddings = []
labels = []
data_dir = '../embeddings/'

for file in os.listdir(data_dir):
    if file.endswith("_embeddings.pkl"):
        class_name = file.split('_')[0]  # Get the class label from filename
        with open(os.path.join(data_dir, file), 'rb') as f:
            class_embeddings = pickle.load(f)
            embeddings.extend(class_embeddings)
            labels.extend([class_name] * len(class_embeddings))

# Set up Annoy index
dimension = len(embeddings[0])  # Embedding dimension
ann_index = AnnoyIndex(dimension, metric='euclidean')

# Add embeddings to the Annoy index
for i, embedding in enumerate(embeddings):
    ann_index.add_item(i, embedding)
ann_index.build(10)  # Build with 10 trees (higher numbers increase precision at the cost of speed)

# Main layout for file upload, k selection, and run button
uploaded_file = st.file_uploader("Choose an X-ray image...", type="png")

# k selection slider in the main column
k = st.slider("Select number of neighbors (k)", min_value=1, max_value=50, value=20)

# Function to classify a new image using Annoy
def classify_image(image, k):
    embedding = generate_embedding(image)
    neighbor_ids = ann_index.get_nns_by_vector(embedding, k)

    # Retrieve labels for the nearest neighbors
    neighbor_labels = [labels[i] for i in neighbor_ids]

    # Count neighbor labels for majority vote and display counts
    label_counts = Counter(neighbor_labels)
    majority_label = label_counts.most_common(1)[0][0]
    return majority_label, label_counts

# Run button to trigger classification
if uploaded_file and st.button("Run Classification"):
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Classify the uploaded image
    predicted_class, neighbor_counts = classify_image(image, k)

    # Display the prediction in a highlighted box
    pretty_class = class_labels.get(predicted_class, predicted_class)  # Convert to pretty name
    st.markdown(f"## Predicted Class: **{pretty_class}**")

    # Display neighbor class counts in an organized layout
    st.markdown("### Neighbor Class Counts:")
    for label, count in neighbor_counts.items():
        pretty_label = class_labels.get(label, label)
        st.write(f"- **{pretty_label}**: {count} neighbors")
