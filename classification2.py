## Classification problem 
'''
Binary classification 
2 circles: 
x^2 + y^2 = 1  (0)
x^2 + y^2 = 2  (1)
'''

import torch 
from torch import nn 
import random, sys 

class BinaryClassificationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 3), 
            nn.ReLU(),
            nn.Linear(3, 1), 
        )
    
    def forward(self, x: torch.Tensor): 
        return torch.sigmoid(self.model(x))

def generate_dataset():
    X_list = []
    Y_list = []
    for _ in range(100):
        x1 = random.randint(-100, 100) / 100
        y = 1
        x2 = (y**2 - x1**2) ** 0.5
        if random.randint(1, 2) == 1: x2 = -x2

        # x1 += random.gauss(0, 0.2)
        # x2 += random.gauss(0, 0.2)

        X_list.append([x1, x2])
        Y_list.append(float(0.0))

    for _ in range(100):
        x1 = random.randint(-200, 200) / 100
        y = 2
        x2 = (y**2 - x1**2) ** 0.5
        if random.randint(1,2) == 1: x2 = -x2

        # x1 += random.gauss(0, 0.2)
        # x2 += random.gauss(0, 0.2)

        X_list.append([x1, x2])
        Y_list.append(float(1.0))
    
    return X_list, Y_list 

def main():
    X_list, Y_list = generate_dataset()
    
    model = BinaryClassificationModel()
    epochs = 20000
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(params=model.parameters(), lr=0.01)

    choices = [i for i in range(200)]
    random.shuffle(choices)
    X_train = torch.tensor([X_list[c] for c in choices[:150]])
    Y_train = torch.tensor([Y_list[c] for c in choices[:150]]).squeeze()
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train).squeeze()   
        loss = loss_fn(y_pred, Y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 500 == 0:
            print(f"epoch: {epoch} | loss: {loss}")

    X_test = [X_list[c] for c in choices[150:]]
    Y_test = [Y_list[c] for c in choices[150:]]

    model.eval()
    with torch.inference_mode():
        accuracy = 0
        total = 0
        for i in range(len(X_test)):
            x_val = torch.tensor(X_test[i])
            y_val = torch.tensor(Y_test[i])
            y_pred = torch.round(model(x_val))
            if y_pred == y_val: accuracy += 1
            total += 1

            if random.randint(1, 5) == 1: 
                print(f"x_val: [{x_val[0]}, {x_val[1]}]\ny_val: {y_val} | y_pred: {y_pred.item()}")
        
        print(accuracy/total)
        print(list(model.parameters()))

if __name__ == '__main__':
    main()