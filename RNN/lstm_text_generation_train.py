import re
import torch
import torch.nn as nn

from collections import Counter

from torch.utils.data import Dataset, DataLoader

# -----------------------------------
# Device
# -----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------------
# Load Dataset
# -----------------------------------
with open(
    "datasets/shakespeare.txt",
    "r",
    encoding="utf-8"
) as file:

    text_data = file.read()

# -----------------------------------
# Clean Text
# -----------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    return text

text_data = clean_text(text_data)

# -----------------------------------
# Tokenization
# -----------------------------------
all_words = text_data.split()

# -----------------------------------
# Vocabulary
# -----------------------------------
counter = Counter(all_words)

# Limit vocabulary size
most_common_words = counter.most_common(3000)

vocab = {

    word: idx + 1

    for idx, (word, freq)

    in enumerate(most_common_words)
}

# Padding token
vocab["<PAD>"] = 0

# Reverse Vocabulary
index_to_word = {

    idx: word

    for word, idx

    in vocab.items()
}

# -----------------------------------
# Create Sequences
# -----------------------------------
sequence_length = 5

input_sequences = []

target_words = []

for i in range(
    sequence_length,
    len(all_words)
):

    seq = all_words[
        i-sequence_length:i
    ]

    target = all_words[i]

    encoded_seq = [

        vocab.get(word, 0)

        for word in seq
    ]

    encoded_target = vocab.get(
        target,
        0
    )

    # Skip PAD targets
    if encoded_target == 0:

        continue

    input_sequences.append(
        encoded_seq
    )

    target_words.append(
        encoded_target
    )

print(
    f"Total Sequences: "
    f"{len(input_sequences)}"
)

# -----------------------------------
# Dataset Class
# -----------------------------------
class TextDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (

            torch.tensor(
                self.X[idx],
                dtype=torch.long
            ),

            torch.tensor(
                self.y[idx],
                dtype=torch.long
            )
        )

# -----------------------------------
# DataLoader
# -----------------------------------
dataset = TextDataset(
    input_sequences,
    target_words
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

# -----------------------------------
# LSTM Model
# -----------------------------------
class LSTMGenerator(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim):

        super(
            LSTMGenerator,
            self
        ).__init__()

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

    def forward(self, x):

        embedded = self.embedding(x)

        outputs, (hidden, cell) = self.lstm(
            embedded
        )

        out = self.fc(
            hidden[-1]
        )

        return out

# -----------------------------------
# Initialize Model
# -----------------------------------
model = LSTMGenerator(

    vocab_size=len(vocab) + 1,

    embedding_dim=128,

    hidden_dim=128

).to(device)

# -----------------------------------
# Loss and Optimizer
# -----------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# Training
# -----------------------------------
epochs = 5

print("\nTraining Started...\n")

for epoch in range(epochs):

    running_loss = 0.0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(

        f"Epoch [{epoch+1}/{epochs}] "

        f"Loss: "
        f"{running_loss/len(loader):.4f}"
    )

print("\nTraining Finished!")

# -----------------------------------
# Save Model
# -----------------------------------
torch.save(
    model.state_dict(),
    "models/lstm_text_generator.pth"
)

torch.save(
    vocab,
    "models/lstm_vocab.pth"
)

torch.save(
    index_to_word,
    "models/lstm_index_to_word.pth"
)

print("\nModel Saved!")