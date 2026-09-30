import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------------
# Model Architecture
# -----------------------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.reshape(-1, 28 * 28)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():

    model = ANN()

    model.load_state_dict(
        torch.load(
            "models/ann_mnist.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

# -----------------------------------
# Main Function
# -----------------------------------
def run():

    st.title("MNIST Handwritten Digit Recognition")

    st.write("Upload a digit image.")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        st.image(
            image,
            caption="Uploaded Image",
            width=150
        )

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            outputs = model(image_tensor)

            _, prediction = torch.max(outputs, 1)

        st.success(f"Predicted Digit: {prediction.item()}")