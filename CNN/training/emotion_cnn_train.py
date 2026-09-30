import warnings
warnings.filterwarnings("ignore")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------
# Device
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# -----------------------------------
# Hyperparameters
# -----------------------------------
batch_size = 32
learning_rate = 0.001
epochs = 10

# -----------------------------------
# Transform
# -----------------------------------
transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((48, 48)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------------
# Dataset
# -----------------------------------
train_dataset = datasets.ImageFolder(
    root="datasets/FER2013/train",
    transform=transform
)


test_dataset = datasets.ImageFolder(
    root="datasets/FER2013/test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# -----------------------------------
# Emotion Classes
# -----------------------------------
print("\nClasses:")
print(train_dataset.classes)

# -----------------------------------
# CNN Model
# -----------------------------------
class EmotionCNN(nn.Module):

    def __init__(self):

        super(EmotionCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, 7)
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x

model = EmotionCNN().to(device)

# -----------------------------------
# Loss & Optimizer
# -----------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# -----------------------------------
# Training
# -----------------------------------
print("\nTraining Started...\n")

for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

print("\nTraining Finished!")

# -----------------------------------
# Testing Accuracy
# -----------------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# -----------------------------------
# Save Model
# -----------------------------------
torch.save(
    model.state_dict(),
    "models/emotion_cnn.pth"
)

print("\nModel Saved!")