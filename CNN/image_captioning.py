import streamlit as st

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms

# -----------------------------------
# Load Vocabulary
# -----------------------------------
vocab = torch.load(
    "models/image_caption_vocab.pth"
)

index_to_word = torch.load(
    "models/image_caption_index_to_word.pth"
)

# -----------------------------------
# Parameters
# -----------------------------------
max_length = 20

# -----------------------------------
# Transform
# -----------------------------------
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------
# Decoder Model
# -----------------------------------
class DecoderRNN(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim):

        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, captions):

        embedded = self.embedding(captions)

        outputs, _ = self.lstm(embedded)

        outputs = self.fc(outputs)

        return outputs

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():

    model = DecoderRNN(
        vocab_size=len(vocab),
        embedding_dim=256,
        hidden_dim=256
    )

    model.load_state_dict(
        torch.load(
            "models/image_caption_decoder.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

# -----------------------------------
# Caption Generation
# -----------------------------------
def generate_caption(model):

    current_seq = [vocab["<START>"]]

    for _ in range(max_length):

        input_tensor = torch.tensor(
            current_seq,
            dtype=torch.long
        ).unsqueeze(0)

        with torch.no_grad():

            outputs = model(input_tensor)

        predicted_id = outputs[0, -1].argmax().item()

        predicted_word = index_to_word.get(
            predicted_id,
            "<UNK>"
        )

        if predicted_word == "<END>":
            break

        current_seq.append(predicted_id)

    words = []

    for idx in current_seq:

        word = index_to_word.get(idx, "")

        if word not in ["<START>", "<END>", "<PAD>"]:

            words.append(word)

    return " ".join(words)

# -----------------------------------
# Main Function
# -----------------------------------
def run():

    st.title("Image Captioning")

    st.write(
        "Upload an image to generate caption."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Transform image
        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        # Generate caption
        caption = generate_caption(model)

        st.success(
            f"Generated Caption: {caption}"
        )