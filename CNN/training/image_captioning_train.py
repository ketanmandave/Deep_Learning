import os
import re
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image

from collections import Counter

from torchvision import transforms, models

from torch.utils.data import Dataset, DataLoader

# -----------------------------------
# Device
# -----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------------
# Load Captions
# -----------------------------------
captions_file = "datasets/Flickr8k/captions.txt"

df = pd.read_csv(captions_file)

print(df.head())

# -----------------------------------
# Clean Captions
# -----------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

df["caption"] = df["caption"].apply(clean_text)

# -----------------------------------
# Add START and END Tokens
# -----------------------------------
df["caption"] = (
    "<START> "
    + df["caption"]
    + " <END>"
)

# -----------------------------------
# Build Vocabulary
# -----------------------------------
counter = Counter()

for caption in df["caption"]:

    counter.update(caption.split())

vocab = {
    word: idx + 1
    for idx, (word, freq)
    in enumerate(counter.items())
}

vocab["<PAD>"] = 0

index_to_word = {
    idx: word
    for word, idx in vocab.items()
}

# -----------------------------------
# Parameters
# -----------------------------------
max_length = 20

# -----------------------------------
# Image Transform
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
# Dataset Class
# -----------------------------------
class FlickrDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        image_name = self.df.iloc[idx, 0]

        caption = self.df.iloc[idx, 1]

        image_path = os.path.join(
            "datasets/Flickr8k/Images",
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        tokens = caption.split()

        sequence = [
            vocab[token]
            for token in tokens
        ]

        if len(sequence) < max_length:

            sequence += [0] * (
                max_length - len(sequence)
            )

        else:

            sequence = sequence[:max_length]

        input_seq = torch.tensor(
            sequence[:-1],
            dtype=torch.long
        )

        target_seq = torch.tensor(
            sequence[1:],
            dtype=torch.long
        )

        return image, input_seq, target_seq

dataset = FlickrDataset(df)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# -----------------------------------
# Encoder
# -----------------------------------
mobilenet = models.mobilenet_v2(
    pretrained=True
)

encoder = mobilenet.features

for param in encoder.parameters():
    param.requires_grad = False

encoder = encoder.to(device)

# -----------------------------------
# Decoder
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

decoder = DecoderRNN(
    vocab_size=len(vocab),
    embedding_dim=256,
    hidden_dim=256
).to(device)

# -----------------------------------
# Loss & Optimizer
# -----------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    decoder.parameters(),
    lr=0.001
)

# -----------------------------------
# Training
# -----------------------------------
epochs = 5

print("\nTraining Started...\n")

for epoch in range(epochs):

    running_loss = 0.0

    for images, input_seq, target_seq in loader:

        input_seq = input_seq.to(device)

        target_seq = target_seq.to(device)

        outputs = decoder(input_seq)

        outputs = outputs.reshape(
            -1,
            len(vocab)
        )

        target_seq = target_seq.reshape(-1)

        loss = criterion(
            outputs,
            target_seq
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(loader):.4f}"
    )

print("\nTraining Finished!")

# -----------------------------------
# Save Models
# -----------------------------------
torch.save(
    decoder.state_dict(),
    "models/image_caption_decoder.pth"
)

torch.save(
    vocab,
    "models/image_caption_vocab.pth"
)

torch.save(
    index_to_word,
    "models/image_caption_index_to_word.pth"
)

print("\nModel Saved!")