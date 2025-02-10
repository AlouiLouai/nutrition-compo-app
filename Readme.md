# Food Nutrition Estimation Model

## Overview
This repository contains a deep learning model that estimates the nutritional composition (calories, fat, carbohydrates, and protein) 
of food items from images. The model is built using PyTorch and is trained on the **mmathys/food-nutrients** dataset from Hugging Face.

## Dataset
The dataset used is from Hugging Face: [mmathys/food-nutrients](https://huggingface.co/datasets/mmathys/food-nutrients). 
It contains food images along with their corresponding nutritional values.

## Features
- Uses ResNet50 as the base CNN model
- Predicts four nutritional values: **total calories, total fat, total carbohydrates, and total protein**
- Implements data preprocessing and augmentation
- Saves the trained model in **TorchScript** format for deployment

## Installation
To set up the virtual environment and install the required dependencies, run:

```bash
# Create a virtual environment
python -m venv venv
```

```bash
# Activate the virtual environment
# On Windows
./venv/Scripts/activate
# On Linux/macOS
source venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

## Usage
To train and save the model, simply run:

```bash
python model-food.py
```

This will:
1. Load and preprocess the dataset.
2. Train the CNN model.
3. Save the trained model as `nutrition_model.pt` in TorchScript format.
4. Evaluate the model on the test dataset.

## Model Structure
- **`model-food.py`**: Contains the entire pipeline, including data loading, preprocessing, training, evaluation, and model saving.
- **`requirements.txt`**: Lists all the dependencies required for the project.

## Dependencies
The project requires the following Python packages:

```bash
- torch==2.6.0
- torchvision==0.21.0
- transformers==4.48.3
- datasets==3.2.0
- pandas==2.2.3
- numpy==2.0.2
- pillow==11.1.0
- scikit-learn==1.6.1
- tqdm==4.67.1
```

## Output
After running `model-food.py`, the trained model will be saved as `nutrition_model.pt`.

## License
This project is open-source and available for public use.
```

Contribution are welcome ! 🚀

