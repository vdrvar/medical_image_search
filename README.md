# Medical Image Similarity Search Engine

This project is a medical image similarity search engine built using **Rust** and **Python**. It allows medical professionals to upload medical images (such as X-rays or MRIs) and retrieve visually similar images from a database to assist in diagnosis.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Design Decisions](#design-decisions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Image Upload**: Users can upload medical images via a web interface.
- **Image Embedding Generation**: Generates image embeddings using pre-trained convolutional neural networks (e.g., ResNet).
- **Efficient Similarity Search**: Utilizes an Approximate Nearest Neighbors (ANN) algorithm (e.g., HNSW) for efficient retrieval of similar images.
- **Result Presentation**: Displays the top-k most similar images along with their classifications.
- **Diagnosis Assistance**: Employs majority voting among retrieved images to suggest a diagnosis.

## Project Structure

- **Python Scripts**:
  - `generate_embeddings.py`: Generates embeddings for images using pre-trained models and saves them in pickle files.
- **Rust Components**:
  - **ANN Implementation**: Loads embeddings and performs similarity search using efficient ANN algorithms.
  - **Web Frontend**: Built using the Rocket framework, allows users to upload images and view results.
- **Data Directory (`data/`)**:
  - Contains subdirectories for each category (e.g., `COVID`, `Normal`, `Lung_Opacity`, `Viral Pneumonia`), each with images and metadata.

## Dataset

We use the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data), which includes:

- **COVID-19**: X-ray images of patients with COVID-19.
- **Normal**: X-ray images of healthy patients.
- **Lung Opacity**: Images with lung opacity.
- **Viral Pneumonia**: X-ray images of patients with viral pneumonia.

## Installation

### Prerequisites

- **Rust**: Install Rust from [rust-lang.org](https://www.rust-lang.org/tools/install).
- **Python 3.6+**: Install Python from [python.org](https://www.python.org/downloads/).
- **Python Packages**: Install required Python packages.

### Steps

1. **Clone the Repository**

   bash
   '''
   git clone https://github.com/yourusername/medical-image-search.git
   cd medical-image-search
   '''
   
2. **Set Up Python Environment**

   Navigate to the directory containing Python scripts:

   bash
   '''
   cd python_scripts
   '''

   Install the required Python packages:

   bash
   '''
   pip install -r requirements.txt
   '''

3. **Generate Image Embeddings**

   Run the embedding generation script:

   bash
   '''
   python generate_embeddings.py
   '''

   This will process images in the `data/` directory and save the embeddings to `embeddings.pkl`.

4. **Build the Rust Project**

   Return to the root directory and build the Rust application:

   bash
   '''
   cd ..
   cargo build --release
   '''

## Usage

1. **Run the Server**

   Start the Rust server:

   bash
   '''
   cargo run --release
   '''

2. **Access the Web Interface**

   Open your web browser and go to `http://localhost:8000`.

3. **Upload an Image**

   Use the interface to upload a medical image (X-ray or MRI).

4. **View Results**

   The application will display the top-k most similar images from the database and provide a suggested diagnosis based on majority voting.

## Design Decisions

### Why Use Python for Embedding Generation?

Python has extensive libraries and support for deep learning models, such as PyTorch and TensorFlow, which makes it convenient for processing images and generating embeddings with pre-trained models.

### Why Use Rust for ANN and Frontend?

Rust provides high performance and safety, which is beneficial for implementing efficient ANN algorithms and ensuring a reliable web server. The Rocket framework in Rust allows for building fast and secure web applications.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   bash
   '''
   git checkout -b feature/your-feature-name
   '''

3. **Commit Your Changes**

   bash
   '''
   git commit -am 'Add some feature'
   '''

4. **Push to the Branch**

   bash
   '''
   git push origin feature/your-feature-name
   '''

5. **Submit a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **Datasets**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data)
- **Python Libraries**: PyTorch, NumPy, SciPy
- **Rust Crates**: tch-rs, Rocket, hnsw-rs
- **Inspiration**: Open-source medical imaging projects and the need for efficient diagnostic tools.
