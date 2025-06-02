## Classification problem 


import torch 
from torch import nn 
import random, sys 

class BinaryClassificationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 5), 
            nn.ReLU(),
            nn.Linear(5, 1),
        )
    
    def forward(self, x: torch.Tensor): 
        return torch.sigmoid(self.model(x))

def generate_dataset():
    X_list = []
    Y_list = []
    for i in range(500):
        x1 = random.randint(0, 500) / 50
        y = 10
        x2 = (y**2 - x1**2) ** 0.5

        x1 += random.gauss(0, 1)
        x2 += random.gauss(0, 1)

        X_list.append([x1, x2])
        Y_list.append(float(0.0))

    for i in range(500):
        x1 = random.randint(0, 1000) / 50
        y = 20
        x2 = (y**2 - x1**2) ** 0.5

        x1 += random.gauss(0, 1)
        x2 += random.gauss(0, 1)

        X_list.append([x1, x2])
        Y_list.append(float(1.0))
    
    return X_list, Y_list 

def main():
    X_list, Y_list = generate_dataset()
    
    model = BinaryClassificationModel()
    epochs = 10000
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.SGD(params=model.parameters(), lr=0.1)

    choices = [i for i in range(1000)]
    random.shuffle(choices)
    X_train = torch.tensor([X_list[c] for c in choices[:500]])
    Y_train = torch.tensor([Y_list[c] for c in choices[:500]]).squeeze()
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train).squeeze()   
        loss = loss_fn(y_pred, Y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 1000 == 0:
            print(f"epoch: {epoch} | loss: {loss}")

    X_test = [X_list[c] for c in choices[800:]]
    Y_test = [Y_list[c] for c in choices[800:]]

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

            if random.randint(1, 10) == 1: 
                print(f"x_val: [{x_val[0]}, {x_val[1]}]\ny_val: {y_val} | y_pred: {y_pred.item()}")
        
        print(accuracy/total)
        # print(list(model.parameters()))

if __name__ == '__main__':
    main()