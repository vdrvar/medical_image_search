# Medical Image Classification App

This project is a simple web application that classifies lung X-ray images into four categories: **COVID-19**, **Lung Opacity**, **Normal**, and **Viral Pneumonia**. The app uses a pre-trained ResNet50 model to extract features, and approximate nearest neighbors (ANN) with [Annoy](https://github.com/spotify/annoy) for fast similarity search. The app allows you to upload an X-ray image, adjust the number of neighbors (k) for classification, and view the predicted class.

## Dataset

We use the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) to train and test this model. The dataset includes X-ray images from the following categories:

- **COVID-19**
- **Lung Opacity**
- **Normal**
- **Viral Pneumonia**

The project assumes you download the 4 folders with images from Kaggle and place them in a designated folder called `data\`.

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/medical-image-classification.git
    cd medical-image-classification
    ```

2. **Install dependencies with Poetry**:
    This project uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed.

    ```bash
    # Install dependencies
    poetry install
    ```

3. **Activate the virtual environment**:
    ```bash
    poetry shell
    ```

4. **Generate Embeddings**:
    Run the script to generate embeddings for each class, skipping the `test` folder.

    ```bash
    python scripts/calculate_embeddings.py
    ```

    This will save embeddings for each category in the `embeddings/` folder.

## Running the App

1. **Start the Streamlit app**:
    ```bash
    streamlit run app/app.py
    ```

2. **Using the App**:
    - Upload an X-ray image in PNG format.
    - Adjust the number of neighbors (k) using the slider.
    - Click the "Run Classification" button to get a prediction and view class counts.

## Screenshots

![image](https://github.com/user-attachments/assets/5ef31049-d4cd-4c35-a188-fc063fe8be81)


## License

This project is licensed under the MIT License.
