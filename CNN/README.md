# CNN Image Classification using CIFAR-10

This project is a Deep Learning Computer Vision project built with PyTorch.

The goal is to classify images from the CIFAR-10 dataset into one of 10 object categories using a Convolutional Neural Network (CNN).

## Project Overview

The CIFAR-10 dataset contains:

- 50,000 training images
- 10 object classes
- RGB images
- Image size: 32 × 32 pixels

The model learns visual features directly from the images using convolution and pooling layers.

## CIFAR-10 Classes

The dataset contains the following 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib
- Jupyter Notebook

## Workflow

1. Import PyTorch and Torchvision libraries
2. Load the CIFAR-10 dataset
3. Convert images to tensors
4. Normalize image values
5. Create training and testing DataLoaders
6. Build a CNN architecture
7. Train the model using Cross-Entropy Loss
8. Track training and validation loss
9. Save the best model based on validation loss
10. Evaluate model accuracy on the test dataset

## Data Preprocessing

The images are converted into PyTorch tensors and normalized.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

`ToTensor()` converts image pixel values into tensors.

Normalization transforms the image values approximately from:

```text
[0, 1] → [-1, 1]
```

## Dataset Loading

```python
trainset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
```

The training dataset contains 50,000 images.

## DataLoaders

```python
trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

testloader = DataLoader(
    testset,
    batch_size=64
)
```

The DataLoader divides the dataset into batches for efficient training.

## CNN Architecture

The CNN contains three convolution blocks followed by fully connected layers.

```text
Input Image
   ↓
Conv2D: 3 → 32
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2D: 32 → 64
   ↓
ReLU
   ↓
MaxPool
   ↓
Conv2D: 64 → 128
   ↓
ReLU
   ↓
MaxPool
   ↓
Flatten
   ↓
Linear: 4×4×128 → 256
   ↓
ReLU
   ↓
Linear: 256 → 10
   ↓
Output Classes
```

PyTorch implementation:

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
```

## Training Configuration

- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Weight Decay: `0.001`
- Batch Size: `64`
- Epochs: `10`

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    weight_decay=0.001
)
```

## Training

The model is trained for 10 epochs.

Training loss and validation loss are recorded after every epoch.

```python
train_losses = []
val_losses = []

best_val_loss = float("inf")
```

During training:

```python
for images, labels in trainloader:
    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()
    optimizer.step()
```

## Best Model Saving

The validation loss is monitored during training.

Whenever the validation loss improves, the model weights are saved.

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(
        model.state_dict(),
        "best_model.pt"
    )
```

This allows the best-performing model to be stored instead of relying only on the final epoch.

## Training and Validation Loss

The notebook tracks both training and validation loss.

```python
plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.show()
```

The validation loss reached its lowest value around epoch 9.

## Result

The CNN achieved:

**Test Accuracy: 76.95%**

The model successfully learned visual patterns from the CIFAR-10 dataset using convolutional layers.

## What I Learned

This project helped me understand:

- Convolutional Neural Networks
- Image tensors
- RGB image channels
- Image normalization
- Convolution filters
- Feature maps
- ReLU activation
- Max pooling
- Flattening
- Fully connected layers
- Multi-class classification
- Cross-Entropy Loss
- Adam optimizer
- Weight decay
- Training and validation loss
- Saving the best model
- Evaluating image classification models

## Project Structure

```text
cifar10-image-classification/
│
├── data/
├── cifar10_cnn.ipynb
├── best_model.pt
└── README.md
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd cifar10-image-classification
```

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib jupyter
```

### 3. Open the Notebook

```bash
jupyter notebook
```

Then run:

```text
cifar10_cnn.ipynb
```

The CIFAR-10 dataset will be downloaded automatically by Torchvision.

## Note

This project was created as part of my Deep Learning learning journey to understand how Convolutional Neural Networks can be used for multi-class image classification.

## Author

Ketan Mandave  
B.Tech — Artificial Intelligence & Machine Learning
