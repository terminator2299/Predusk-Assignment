import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import load_data
from model import SimpleNN

# Hyperparameters
input_dim = 4
hidden_dim = 10
output_dim = 3
lr = 0.01
epochs = 50

# Data
X_train, X_test, y_train, y_test = load_data()

# Model, loss, optimizer
model = SimpleNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

train_accuracies = []
test_accuracies = []

# Training
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_pred = torch.argmax(model(X_train), dim=1)
        test_pred = torch.argmax(model(X_test), dim=1)
        train_acc = (train_pred == y_train).float().mean().item()
        test_acc = (test_pred == y_test).float().mean().item()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# Final test accuracy
print(f"\nFinal Test Accuracy: {test_accuracies[-1]:.4f}")

# Plot
plt.plot(range(1, epochs+1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs+1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.savefig("accuracy_plot.png")
plt.show()
