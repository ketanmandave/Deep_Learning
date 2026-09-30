# ANN Fruit Classification

This project is a beginner-friendly Deep Learning classification project built with PyTorch.

The goal is to classify date fruits into one of 7 classes using numerical features extracted from the fruit samples.

## Project Overview

The dataset contains 898 samples and 35 columns.

The input features describe shape, geometry, color, texture, and statistical properties of the fruits.

The target column is:

- `Class` — fruit category label

Classes in the dataset:

- BERHI
- DEGLET
- DOKOL
- IRAQI
- ROTANA
- SAFAVI
- SOGAY

## Tech Stack

- Python
- Pandas
- NumPy
- PyTorch
- Scikit-learn
- Matplotlib

## Workflow

1. Load the dataset
2. Check for missing values
3. Separate input features and target labels
4. Encode class labels using `LabelEncoder`
5. Split the data into training and testing sets
6. Standardize features using `StandardScaler`
7. Convert data into PyTorch tensors
8. Create `TensorDataset` and `DataLoader`
9. Build an Artificial Neural Network
10. Train the model
11. Evaluate classification accuracy

## Data Preprocessing

The target labels are encoded using `LabelEncoder`.

The dataset is split into:

- 80% training data
- 20% testing data

Numerical features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaled data is then converted into PyTorch tensors.

## ANN Architecture

The neural network used in this project has the following structure:

```text
Input Layer
    ↓
Linear(input_size → 64)
    ↓
ReLU
    ↓
Linear(64 → 64)
    ↓
ReLU
    ↓
Linear(64 → 7)
    ↓
Output Classes
```

PyTorch implementation:

```python
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        return self.model(x)
```

## Training Configuration

- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning Rate: `0.001`
- Epochs: `100`
- Batch Size: `32`

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

## Result

The trained ANN achieved:

**Test Accuracy: 96.67%**

The training loss decreased from approximately **1.70** in the first epoch to approximately **0.026** after 100 epochs.

## What I Learned

This project helped me understand:

- How an Artificial Neural Network works
- Multi-class classification
- Forward propagation
- Backpropagation
- ReLU activation
- Cross-entropy loss
- Adam optimizer
- Feature scaling
- Batch training using `DataLoader`
- Model evaluation using test accuracy

## Project Structure

```text
fruit-classification/
│
├── DateFruit_Dataset.csv
├── fruit_classification.ipynb
└── README.md
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd fruit-classification
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib torch scikit-learn
```

### 3. Open the Notebook

```bash
jupyter notebook
```

Then run the cells in `fruit_classification.ipynb`.

## Note

This project was created as part of my Deep Learning learning journey to understand how Artificial Neural Networks can be applied to a real multi-class classification problem.

## Author

Ketan Mandave  
B.Tech — Artificial Intelligence & Machine Learning
