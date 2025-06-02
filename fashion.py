# learning to classify clothes
# from the FashionMNIST Dataset 

import torch 
from torch import nn 
from torch.utils.data import DataLoader

import torchvision 
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import random

DOWNLOAD_PATH = 'C:\\Users\\GAdmin\\Documents\\pythonprojects\\learning_pytorch\\fashion_data'
train_data = datasets.FashionMNIST(
    root=DOWNLOAD_PATH,
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.FashionMNIST(
    root=DOWNLOAD_PATH,
    train=False, 
    download=True,
    transform=ToTensor(),
    target_transform=None
)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

class ConvolutionalModel(nn.Module):
    def __init__(self, input_layers, hidden_layers, output_shape):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_layers, 
                               out_channels=hidden_layers,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layers,
                      out_channels=hidden_layers,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_layers, hidden_layers, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_layers, hidden_layers, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(p=0.1),
            nn.Linear(in_features=hidden_layers*7*7,
                      out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x

HIDDEN_LAYERS = 17

def main():
    epochs = 7

    model = ConvolutionalModel(1, HIDDEN_LAYERS, 10)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)


    for epoch in range(epochs):
        model.train()
        print(f'Epoch: {epoch}')

        for batch, (X, y) in enumerate(train_dataloader):
            
            
            y_pred = model(X) # remember, CrossEntropyLoss needs raw logits
            loss = loss_fn(y_pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if (random.randint(1,1000)==1): print(f"loss: {loss}")

    path_to_save = "C:\\Users\\GAdmin\\Documents\\pythonprojects\\fashion_model.pth"
    torch.save(model.state_dict(), path_to_save)

if __name__ == '__main__': main()

