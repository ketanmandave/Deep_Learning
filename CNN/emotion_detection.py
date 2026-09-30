import streamlit as st
import torch
import torch.nn as nn

from PIL import Image

import torchvision.transforms as transforms

# -----------------------------------
# Emotion Labels
# -----------------------------------
emotion_classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

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

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():

    model = EmotionCNN()

    model.load_state_dict(
        torch.load(
            "models/emotion_cnn.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

# -----------------------------------
# Main Function
# -----------------------------------
def run():

    st.title("Facial Emotion Detection")

    st.write("Upload a face image.")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        transform = transforms.Compose([

            transforms.Resize((48, 48)),

            transforms.ToTensor(),

            transforms.Normalize((0.5,), (0.5,))
        ])

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        emotion = emotion_classes[predicted.item()]

        st.success(
            f"Predicted Emotion: {emotion}"
        )

        st.info(
            f"Confidence: {confidence.item()*100:.2f}%"
        )