import torch 
from torch import nn 
import random, sys

class SimpleLinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.model(x)

def main():
    X_list = torch.arange(0, 100, 1, dtype=torch.float)
    errors = torch.tensor([random.gauss(0,0.5) for _ in range(100)])
    Y_list = 15*X_list + 90
    Y_list = Y_list + errors

    X_train = X_list[:80].unsqueeze(1)
    Y_train = Y_list[:80].unsqueeze(1)
    model = SimpleLinearModel()
    loss_fn = nn.L1Loss()
    optimiser = torch.optim.SGD(params=list(model.parameters()), lr=1e-2)  # Increased learning rate
    epochs = 50000

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, Y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if epoch % 1000 == 0:
            print(f"epoch: {epoch} | loss: {loss}")
    
    print(list(model.parameters()))

if __name__ == '__main__':
    main()