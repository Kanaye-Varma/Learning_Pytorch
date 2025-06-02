import torch 
from torch import nn 
import sys, random
from sklearn.datasets import make_blobs

def generate_datasets(num_classes):
    
    X_values = []
    Y_values = []

    # create num_classes squares of different random radii 

    for i in range(num_classes):

        y = 10*(i+1) + random.randint(0, 2)

        for _ in range(random.randint(100, 150)):
            x1 = random.randint(0, 100*y) / 100
            x2 = (y**2 - x1**2) ** 0.5
            x1 += random.gauss(0, 1)
            x2 += random.gauss(0, 1)
            X_values.append([x1, x2])
            Y_values.append(i)
    
    return X_values, Y_values

class MultiClassification(nn.Module):

    def __init__(self, num_classes : int):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.num_classes+5),
            nn.ReLU(), 
            nn.Linear(in_features=self.num_classes+5, out_features=self.num_classes+10),
            nn.ReLU(), 
            nn.Linear(in_features=self.num_classes+10, out_features=self.num_classes+5), 
            nn.ReLU(), 
            nn.Linear(in_features=self.num_classes+5, out_features=self.num_classes)
        )

    def forward(self, x):
        
        return torch.softmax(self.model(x), dim=1, dtype=torch.float)
    
def main(num_classes): 

    X_list, Y_list = generate_datasets(num_classes)

    choices = [i for i in range(len(X_list))]
    random.shuffle(choices)
    
    test_limit = int(0.8 * len(choices))
    X_train = torch.tensor([X_list[c] for c in choices[:test_limit]])
    Y_train = torch.tensor([Y_list[c] for c in choices[:test_limit]])

    myModel = MultiClassification(num_classes)

    epochs = 10000
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(myModel.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        myModel.train()
        
        y_pred = myModel(X_train)
        loss = loss_fn(y_pred, Y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (epoch % 100 == 0):
            print(f"Epoch: {epoch} | Loss: {loss}")
            c, t = 0, 0
            for i in range(len(y_pred)):
                if torch.argmax(y_pred[i]) == Y_train[i]: c += 1
                t += 1
            print(f"accuracy = {c/t}")
    


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Correct usage: python multiclassification.py [num_classes]")
        sys.exit()
    else:
        main(int(sys.argv[1]))