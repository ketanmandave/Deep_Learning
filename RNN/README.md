# RNN Sentiment Analysis using IMDB Dataset

This project is a Deep Learning Natural Language Processing project built with PyTorch.

The goal is to classify movie reviews from the IMDB dataset as either Positive or Negative using a Recurrent Neural Network (RNN).

## Project Overview

The IMDB dataset originally contains 50,000 movie reviews with two sentiment classes:

- Positive
- Negative

After duplicate removal, the dataset contains approximately 49,582 reviews.

This implementation preserves the sequence of words so the RNN can process text in order.

## Tech Stack

- Python
- Pandas
- NumPy
- PyTorch
- Scikit-learn
- Regular Expressions
- Jupyter Notebook

## Workflow

1. Load the IMDB dataset
2. Remove duplicate reviews
3. Clean review text
4. Encode sentiment labels
5. Split the dataset into training and testing sets
6. Build a vocabulary from training data
7. Convert words into integer sequences
8. Pad or truncate reviews to a fixed length
9. Create PyTorch Dataset and DataLoader objects
10. Build an Embedding + RNN model
11. Train the model
12. Evaluate classification performance
13. Test the model using custom reviews

## Text Preprocessing

The preprocessing includes:

- Converting text to lowercase
- Removing HTML tags
- Removing URLs
- Removing unwanted special characters
- Removing extra spaces

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Stopwords are not aggressively removed because words such as `not` can carry important sentiment information.

## Sentiment Encoding

```text
Negative -> 0
Positive -> 1
```

```python
df["sentiment"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})
```

## Train-Test Split

The dataset is split into:

- 80% training data
- 20% testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)
```

## Vocabulary Creation

A vocabulary is built only from the training reviews.

Special tokens:

```text
<PAD> -> 0
<UNK> -> 1
```

The vocabulary size is limited to:

```python
MAX_VOCAB_SIZE = 20000
```

## Sequence Encoding

Each review is converted into a fixed-length sequence of word IDs.

```python
MAX_LENGTH = 200
```

Short reviews are padded and long reviews are truncated.

```text
Review
  ↓
Tokenization
  ↓
Word IDs
  ↓
Padding / Truncation
  ↓
Fixed-Length Sequence
```

## Model Architecture

```text
Input Review
    ↓
Word Indices
    ↓
Embedding Layer
    ↓
RNN
    ↓
Final Hidden State
    ↓
Fully Connected Layer
    ↓
Sentiment Logit
    ↓
Positive / Negative
```

```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        final_hidden = hidden[-1]
        logits = self.fc(final_hidden)
        return logits.squeeze(1)
```

## Training Configuration

- Model: Simple RNN
- Embedding Dimension: 128
- Hidden Size: 128
- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Learning Rate: `0.001`
- Batch Size: `64`
- Epochs: `5`

```python
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

## Evaluation

The model output is converted to probabilities using sigmoid.

```python
probabilities = torch.sigmoid(logits)

predictions = (probabilities >= 0.5).float()
```

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

## Custom Review Prediction

```python
def predict_sentiment(review):
    model.eval()

    cleaned_review = clean_text(review)
    encoded_review = encode_review(cleaned_review)

    input_tensor = torch.tensor(
        encoded_review,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability

    print("Review:")
    print(review)
    print("\nPredicted Sentiment:", sentiment)
    print(f"Confidence: {confidence * 100:.2f}%")
```

Example:

```python
review = "This movie was absolutely amazing. The story was engaging and the acting was excellent."
predict_sentiment(review)
```

## Why RNN Instead of TF-IDF?

TF-IDF converts a review into a fixed feature vector and does not preserve word order.

An RNN is designed to process sequences.

```text
TF-IDF Approach

Review
  ↓
Fixed Feature Vector
  ↓
Classifier
```

```text
RNN Approach

Review
  ↓
Word Sequence
  ↓
Embedding
  ↓
RNN
  ↓
Sentiment
```

## What I Learned

- NLP preprocessing
- Vocabulary creation
- Tokenization
- Padding and truncation
- Word embeddings
- Sequence modeling
- Recurrent Neural Networks
- Hidden states
- Binary classification
- `BCEWithLogitsLoss`
- PyTorch training loops
- Custom text inference
- Difference between bag-of-words and sequential models

## Project Structure

```text
rnn-sentiment-analysis/
│
├── IMDB Dataset.csv
├── sentiment_analysis_rnn.ipynb
├── sentiment_rnn.pt
└── README.md
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rnn-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install pandas numpy torch scikit-learn jupyter
```

### 3. Open the Notebook

```bash
jupyter notebook
```

Then run:

```text
sentiment_analysis_rnn.ipynb
```

## Note

This project was created as part of my Deep Learning learning journey to understand how Recurrent Neural Networks can be applied to sequence-based sentiment classification.

## Author

Ketan Mandave  
B.Tech — Artificial Intelligence & Machine Learning