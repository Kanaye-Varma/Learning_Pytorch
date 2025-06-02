import torch 
from torch import nn
import sys 

class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.gradient = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=True)
        self.intercept = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=True)
    
    def forward(self, x : torch.Tensor):
        return self.gradient * x + self.intercept

def main(desired_gradient, desired_intercept, epochs, learning_rate):
    X_tensor = torch.arange(0, 100, 1)
    Y_tensor = desired_gradient * X_tensor + desired_intercept

    X_training = X_tensor[:80]
    Y_training = Y_tensor[:80]
    X_testing  = X_tensor[80:]
    Y_testing  = Y_tensor[80:]

    mymodel = LinearRegressionModel()

    # training the model 

    loss_fn = nn.L1Loss()
    optimiser = torch.optim.SGD(params=mymodel.parameters(), lr = learning_rate)

    for _ in range(int(epochs)):
        mymodel.train()

        y_pred = mymodel(X_training)
        loss = loss_fn(y_pred, Y_training)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    mymodel.eval()
    
    print(f"y = {float(mymodel.gradient)}x + {float(mymodel.intercept)}")
    print(f"loss: {loss_fn(mymodel(X_testing), Y_testing)}")



if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python linear_regression.py [gradient] [intercept] [epochs] [learning rate]")
        sys.exit()
    
    args = list(map(float, sys.argv[1:]))
    main(args[0], args[1], args[2], args[3])