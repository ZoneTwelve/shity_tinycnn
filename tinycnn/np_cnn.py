import json
import numpy as np
from tqdm import tqdm
import sys
# Load data from the JSON file
dbn = 1 # dataset index
with open(f'../data/dataset_{dbn}/train_data.json', 'r') as file:
    data = json.load(file)
with open(f'../data/dataset_{dbn}/test_data.json', 'r') as file:
    test_data = json.load(file)

# A simple dense layer
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        return np.dot(input, self.weights) + self.bias

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(predicted, target):
    return -np.log(predicted[np.arange(len(predicted)), target])

# The simplified CNN model
class TinyCNN:
    def __init__(self):
        self.layer1 = DenseLayer(4, 10)  # Simplified dense layer

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = relu(self.layer1.forward(x))
        return softmax(x)

# Instantiate the model
model = TinyCNN()

# Training the model
num_epochs = int(sys.argv[1])#50  # Number of epochs for training
learning_rate = float(sys.argv[2])#2e-6

for epoch in range(num_epochs):
    total_loss = 0
    with tqdm(data, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="sample") as t:
        for item in t:
            pattern = np.array(item['pattern']).reshape(1, 4) / 255.0
            label = np.array([item['label']])

            # Forward pass
            outputs = model.forward(pattern)

            # Compute loss
            loss = cross_entropy_loss(outputs, label)
            total_loss += loss

            # Backward pass (Gradient Descent)
            # ... (Implement gradients computation and weights update)

            t.set_postfix(loss=loss)

    average_loss = total_loss / len(data)
    #print(f"Average Loss in Epoch {epoch}: {average_loss}")


# Simplified testing loop
correct = 0
total = 0
for item in test_data:
    pattern = np.array(item['pattern']).reshape(1, 4) / 255.0
    label = np.array([item['label']])
    outputs = model.forward(pattern)
    predicted = np.argmax(outputs, axis=1)
    total += 1
    correct += (predicted == label).sum()

print(f'Accuracy: {100 * correct / total}%')
