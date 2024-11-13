import torch
import torch.nn as nn
import torch.optim as optim


# Beispielhafte Modellklasse (Anpassung nach Bedarf)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # Das ist ja nicht der hidden layer, eher eine 12,5 Ausgabe Batch oder so
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Beispielhafte Trainingsfunktion
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Beispielhafte Testfunktion
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Parameterkombinationen
param_grid = [
    {'input_size': 784, 'hidden_size': 128, 'output_size': 10, 'lr': 0.01, 'num_epochs': 5},
    {'input_size': 784, 'hidden_size': 64, 'output_size': 10, 'lr': 0.001, 'num_epochs': 10},
    # Weitere Parameterkombinationen hinzufügen
]

# Beispielhafte Datenloader (anpassen nach Bedarf)
train_loader =  # Dein train_loader
test_loader =  # Dein test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Schleife über Parameterkombinationen
for params in param_grid:
    print(f"Training mit Parametern: {params}")

    # Modell initialisieren
    model = NeuralNetwork(params['input_size'], params['hidden_size'], params['output_size']).to(device)

    # Verlustfunktion und Optimierer definieren
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])

    # Training
    train_model(model, train_loader, criterion, optimizer, num_epochs=params['num_epochs'])

    # Testen
    accuracy = test_model(model, test_loader)
    print(f"Testgenauigkeit: {accuracy}%\n")

