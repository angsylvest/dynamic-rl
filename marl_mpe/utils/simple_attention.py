# from numpy import array
# from numpy import random
# from numpy import dot
# from scipy.special import softmax

# # encoder representations of four different words
# word_1 = array([1, 0, 0])
# word_2 = array([0, 1, 0])
# word_3 = array([1, 1, 0])
# word_4 = array([0, 0, 1])

# # stacking the word embeddings into a single array
# words = array([word_1, word_2, word_3, word_4])

# # generating the weight matrices
# random.seed(42)
# W_Q = random.randint(3, size=(3, 3))
# W_K = random.randint(3, size=(3, 3))
# W_V = random.randint(3, size=(3, 3))

# print(f'W_Q {W_Q}')

# # generating the queries, keys and values
# Q = words @ W_Q
# K = words @ W_K
# V = words @ W_V

# print(f'Q {Q} \n, K {K} \n, V {V}')

# # scoring the query vectors against all key vectors
# scores = Q @ K.transpose()

# # computing the weights by a softmax operation
# weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# # computing the attention by a weighted sum of the value vectors
# attention = weights @ V

# print(attention)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple toy dataset
# Each input is a sequence of word embeddings, and the output is a sentiment label
# For simplicity, let's assume the sentence has a maximum length of 5
# The sentiment label is 0 for negative and 1 for positive
# Here, we're using random word embeddings for demonstration purposes

# Toy dataset
# Input: Word embeddings for each word in the sentence
# Output: Sentiment label (0 for negative, 1 for positive)
data = [
    (np.random.rand(5, 10), 1),  # Positive example
    (np.random.rand(5, 10), 0),  # Negative example
]

# Define the attention network
class AttentionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionNetwork, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply fully connected layer
        x = torch.relu(self.fc(x))
        # Apply attention mechanism
        attention_scores = self.attention(x)
        attention_weights = self.softmax(attention_scores)
        # Apply weighted sum
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector

# Define the sentiment classifier
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Corrected input_size to hidden_size
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Define the model
class AttentionSentimentModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionSentimentModel, self).__init__()
        self.attention_net = AttentionNetwork(input_size, hidden_size)
        self.sentiment_classifier = SentimentClassifier(hidden_size, hidden_size)

    def forward(self, x):
        context_vector = self.attention_net(x)
        sentiment_score = self.sentiment_classifier(context_vector)
        return sentiment_score

# Initialize the model
input_size = 10  # Size of word embeddings
hidden_size = 16  # Size of hidden layer
model = AttentionSentimentModel(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for input_data, target in data:
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor([target], dtype=torch.float32)

        # Forward pass
        output = model(input_tensor)
        loss = criterion(output, target_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(data)}")

# Test the trained model
with torch.no_grad():
    for input_data, target in data:
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        predicted_label = 1 if output.item() > 0.5 else 0
        print(f"Predicted Sentiment: {'Positive' if predicted_label == 1 else 'Negative'}, True Sentiment: {'Positive' if target == 1 else 'Negative'}")
