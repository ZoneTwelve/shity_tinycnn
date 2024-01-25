import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# Load data from the JSON file
dbn = 0 # dataset index
with open(f'dataset_{dbn}/train_data.json', 'r') as file:
    data = json.load(file)
with open(f'dataset_{dbn}/test_data.json', 'r') as file:
    test_data = json.load(file)

# Optionally, split the data into training and testing sets
# For simplicity, let's assume the entire data is used for training
# In a real application, you should split the data

# Define the CNN Model
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2)
        self.fc1 = nn.Linear(4, 10)  # Assuming 10 output classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 4)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x

# Instantiate the model
model = TinyCNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 50  # Number of epochs for training
#for epoch in range(num_epochs):
#    print('Epoch:', epoch)
#    for item in data:
for epoch in range(num_epochs):
    total_loss = 0
    with tqdm(data, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="sample") as t:
        for item in t:
            pattern = torch.tensor(item['pattern']).view(-1, 1, 2, 2).float() / 255.0
            label = torch.tensor([item['label']])

            optimizer.zero_grad()
            outputs = model(pattern)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

# Example of how you might test the model (assuming you have a test set)

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for item in test_data:
        pattern = torch.tensor(item['pattern']).view(-1, 1, 2, 2).float() / 255.0
        label = torch.tensor([item['label']])
        outputs = model(pattern)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Accuracy: {100 * correct // total}%')

torch.save(model, 'model.pt')

