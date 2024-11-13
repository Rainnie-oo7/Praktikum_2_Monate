import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# Download training PennFudanPed from open datasets.
training_data = datasets.FashionMNIST(
    root="PennFudanPed",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test PennFudanPed from open datasets.
test_data = datasets.FashionMNIST(
    root="PennFudanPed",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 8

# Create PennFudanPed loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(64 * 3 * 3, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 3 * 3)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, in_channels, out_channels, kernel_size, stride, padding, maxpoolkernel, maxpoolstride, maxpoolpadding):
        super(NeuralNetwork, self).__init__()
    def convolu(self):
        a = 1
        b = 16

        durchlaeufe = 3

        for i in range(durchlaeufe):
            self.methode(a, b) # Aufruf der Methode mit den aktuellen Werten von a und b

            #Neue WErte für a und b
            a = b  # a nimmt den alten Wert von b
            b = b + (b -a+1) # b wird gemäß einer Regel (hier: inkrementieren) aktualisiert

        def methode

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size, stride=2, padding=0)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Das ist mir zu knapp, daher das ^ oben  in ausführlich geschrieben wollen
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# param_grid_init = [
#     {},
#     {},
#     # Weitere Parameterkombinationen hinzufügen
# ]
#
# model = NeuralNetwork().to(device)
# print(model)



# Parameterkombinationen
param_grid = [
    {'input_size': 784, 'hidden_size': 128, 'output_size': 10, 'lr': 0.01, 'num_epochs': 5},
    {'input_size': 784, 'hidden_size': 64, 'output_size': 10, 'lr': 0.001, 'num_epochs': 10},
    # Weitere Parameterkombinationen hinzufügen # Problem funz ja eigentlich nur für EPochen, daher oben neu für das Modul
]

for params in param_grid:
    print(f"Training mit Parametern: {params}")

    # Modell initialisieren
    model = NeuralNetwork(params['input_size'], params['hidden_size'], params['output_size']).to(device)

    # Verlustfunktion und Optimierer definieren
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

    # Training
    train(model, train_dataloader, loss_fn, optimizer, num_epochs=params['num_epochs'])

    # Testen
    accuracy = test(model, test_dataloader)
    print(f"Testgenauigkeit: {accuracy}%\n")


epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[59][0], test_data[59][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')