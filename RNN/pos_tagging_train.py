import nltk
import torch
import torch.nn as nn

from collections import Counter

from nltk.corpus import treebank

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

# -----------------------------------
# Download Dataset
# -----------------------------------
nltk.download('treebank')

# -----------------------------------
# Device
# -----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------------
# Load Tagged Sentences
# -----------------------------------
tagged_sentences = treebank.tagged_sents()

print("Total Sentences:", len(tagged_sentences))

# -----------------------------------
# Build Vocabulary
# -----------------------------------
word_counter = Counter()
tag_counter = Counter()

for sentence in tagged_sentences:

    for word, tag in sentence:

        word_counter[word.lower()] += 1
        tag_counter[tag] += 1

# Word Vocabulary
word_vocab = {
    word: idx + 2
    for idx, word in enumerate(word_counter)
}

word_vocab["<PAD>"] = 0
word_vocab["<UNK>"] = 1

# Tag Vocabulary
tag_vocab = {
    tag: idx
    for idx, tag in enumerate(tag_counter)
}

index_to_tag = {
    idx: tag
    for tag, idx in tag_vocab.items()
}

# -----------------------------------
# Parameters
# -----------------------------------
max_length = 50

# -----------------------------------
# Convert Sentence to Sequence
# -----------------------------------
def encode_sentence(sentence):

    words = []
    tags = []

    for word, tag in sentence:

        words.append(
            word_vocab.get(
                word.lower(),
                1
            )
        )

        tags.append(
            tag_vocab[tag]
        )

    # Padding
    if len(words) < max_length:

        pad_size = max_length - len(words)

        words += [0] * pad_size
        tags += [0] * pad_size

    else:

        words = words[:max_length]
        tags = tags[:max_length]

    return words, tags

encoded_data = [
    encode_sentence(sentence)
    for sentence in tagged_sentences
]

# -----------------------------------
# Split Dataset
# -----------------------------------
train_data, test_data = train_test_split(
    encoded_data,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Dataset Class
# -----------------------------------
class POSDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        words, tags = self.data[idx]

        return (
            torch.tensor(words, dtype=torch.long),
            torch.tensor(tags, dtype=torch.long)
        )

train_dataset = POSDataset(train_data)

test_dataset = POSDataset(test_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# -----------------------------------
# RNN Model
# -----------------------------------
class POSTagger(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 output_dim):

        super(POSTagger, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(self, x):

        embedded = self.embedding(x)

        outputs, hidden = self.rnn(embedded)

        outputs = self.fc(outputs)

        return outputs

model = POSTagger(
    vocab_size=len(word_vocab),
    embedding_dim=128,
    hidden_dim=128,
    output_dim=len(tag_vocab)
).to(device)

# -----------------------------------
# Loss & Optimizer
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

    for words, tags in train_loader:

        words = words.to(device)

        tags = tags.to(device)

        outputs = model(words)

        outputs = outputs.reshape(
            -1,
            len(tag_vocab)
        )

        tags = tags.reshape(-1)

        loss = criterion(
            outputs,
            tags
        )

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
# Save Model
# -----------------------------------
torch.save(
    model.state_dict(),
    "models/pos_tagger.pth"
)

torch.save(
    word_vocab,
    "models/pos_word_vocab.pth"
)

torch.save(
    tag_vocab,
    "models/pos_tag_vocab.pth"
)

torch.save(
    index_to_tag,
    "models/pos_index_to_tag.pth"
)

print("\nModel Saved!")