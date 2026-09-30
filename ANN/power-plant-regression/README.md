# ANN Power Plant Energy Prediction

This project is a Deep Learning regression project built with PyTorch.

The goal is to predict the electrical energy produced by a power plant using four environmental input features.

## Project Overview

The dataset contains **9568 samples** and **5 columns**.

The input features are:

- `AT` — Ambient Temperature
- `V` — Exhaust Vacuum
- `AP` — Ambient Pressure
- `RH` — Relative Humidity

The target column is:

- `PE` — Produced Energy

This is a regression problem because the model predicts a continuous numerical value.

## Tech Stack

- Python
- Pandas
- NumPy
- PyTorch
- Scikit-learn
- Matplotlib

## Workflow

1. Load the power plant dataset
2. Check for missing values
3. Separate input features and target
4. Split the dataset into training and testing sets
5. Standardize the input features using `StandardScaler`
6. Convert data into PyTorch tensors
7. Create `TensorDataset` and `DataLoader`
8. Build an Artificial Neural Network for regression
9. Train the model using Mean Squared Error loss
10. Track training and validation loss
11. Save the best model based on validation loss
12. Evaluate the model using MSE and R² score
13. Compare actual and predicted energy values

## Data Preprocessing

The dataset does not contain missing values.

The data is split into:

- **80% training data**
- **20% testing data**

The four input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)
```

The scaled features and target values are converted into PyTorch tensors.

```python
X_train_tensors = torch.tensor(X_train_scale, dtype=torch.float32)
X_test_tensors = torch.tensor(X_test_scale, dtype=torch.float32)

y_train_tensors = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test_tensors = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)
```

## ANN Architecture

The neural network used in this project has the following structure:

```text
Input Layer (4 features)
        ↓
Linear(4 → 6)
        ↓
ReLU
        ↓
Linear(6 → 6)
        ↓
ReLU
        ↓
Linear(6 → 1)
        ↓
Predicted Energy Output
```

PyTorch implementation:

```python
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        return self.model(x)
```

## Training Configuration

- Loss Function: `MSELoss`
- Optimizer: `Adam`
- Learning Rate: `0.001`
- Epochs: `100`
- Batch Size: `32`

```python
criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

## Best Model Saving

During training, the validation loss is monitored.

Whenever the validation loss improves, the model weights are saved:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")
```

This ensures that the final evaluation uses the best-performing model rather than simply the model from the last epoch.

## Results

The best model achieved:

- **Train MSE:** 21.10
- **Test MSE:** 19.39
- **R² Score:** 0.9322

An R² score of approximately **0.93** means that the model explains about 93% of the variation in the produced energy values on the test data.

## Example Predictions

| Actual Energy | Predicted Energy |
|---:|---:|
| 433.27 | 435.62 |
| 438.16 | 437.23 |
| 458.42 | 461.28 |
| 480.82 | 475.96 |
| 441.41 | 435.87 |

## Training and Validation Loss

Training and validation loss are tracked for every epoch.

```python
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

The loss falls rapidly during the initial epochs and later stabilizes.

## What I Learned

This project helped me understand:

- ANN-based regression
- Difference between classification and regression
- Feature scaling for neural networks
- PyTorch tensors
- Mini-batch training using `DataLoader`
- Forward propagation
- Backpropagation
- ReLU activation
- Mean Squared Error loss
- Adam optimizer
- Training vs validation loss
- Saving the best model
- Evaluating regression models using MSE and R² score
- Comparing actual and predicted values

## Project Structure

```text
power-plant-energy-prediction/
│
├── powerplant_data.csv
├── powerplant_energy_prediction.ipynb
├── best_model.pt
└── README.md
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd power-plant-energy-prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib torch scikit-learn jupyter
```

### 3. Open the Notebook

```bash
jupyter notebook
```

Then run the cells in:

```text
powerplant_energy_prediction.ipynb
```

## Note

This project was created as part of my Deep Learning learning journey to understand how Artificial Neural Networks can be applied to a regression problem.

## Author

Ketan Mandave  
B.Tech — Artificial Intelligence & Machine Learning
