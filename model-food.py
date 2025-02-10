import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import StepLR

# Step 1: Load and Preprocess Data
def load_and_preprocess_data():
    dataset = load_dataset("mmathys/food-nutrients")
    data = dataset['test']
    df = pd.DataFrame(data)
    
    # Normalize nutritional values
    nutritional_columns = ['total_calories', 'total_fat', 'total_carb', 'total_protein']
    df[nutritional_columns] = df[nutritional_columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    return df

# Step 2: Create a Custom Dataset
class FoodDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df.iloc[idx]['image']
        if isinstance(image, dict):
            image = Image.open(image['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        nutrition = self.df.iloc[idx][['total_calories', 'total_fat', 'total_carb', 'total_protein']].values.astype(np.float32)
        return image, nutrition

# Step 3: Define the CNN Model
class NutritionCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)  # Directly initialize the parent class
        # Using the updated method for loading ResNet50 with weights
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Replace the fully connected layer to output 4 values (calories, fat, carb, protein)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 4)
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return x

# Step 4: Train the Model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, nutrition in train_loader:
            images, nutrition = images.to(device), nutrition.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, nutrition)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, nutrition in val_loader:
                images, nutrition = images.to(device), nutrition.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, nutrition).item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# Step 5: Evaluate the Model
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    actuals, predictions = [], []

    with torch.no_grad():
        for images, nutrition in test_loader:
            images, nutrition = images.to(device), nutrition.to(device)
            outputs = model(images)
            actuals.extend(nutrition.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

    print("\nSample Predictions:")
    for i in range(5):
        print(f"Actual: {actuals[i]}, Predicted: {predictions[i]}")

# Step 6: Main Function
def main():
    df = load_and_preprocess_data()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FoodDataset(df, transform=transform)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_dataset = FoodDataset(train_df, transform=transform)
    val_dataset = FoodDataset(val_df, transform=transform)
    test_dataset = FoodDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = NutritionCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

    # Save the model to TorchScript format after training
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save("nutrition_model.pt")
    print("Model saved to nutrition_model.pt")

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
