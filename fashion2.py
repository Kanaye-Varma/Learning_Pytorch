from fashion import ConvolutionalModel, test_dataloader, HIDDEN_LAYERS
import torch 

path_to_load = "C:\\Users\\GAdmin\\Documents\\pythonprojects\\fashion_model.pth"
model = ConvolutionalModel(1, HIDDEN_LAYERS, 10)
model.load_state_dict(torch.load(path_to_load, weights_only=True))
model.eval()



acc_list = []
with torch.inference_mode():
    
    for batch, (X, y) in enumerate(test_dataloader):
        
        y_pred = torch.argmax(torch.softmax(model(X), dim=1), dim=1)
        
        accuracy = 0
        for i in range(len(y)):
            if y_pred[i] == y[i]: accuracy += 1
        
        accuracy /= len(y)
        print(accuracy)
        
        acc_list.append(accuracy)

    
    print(f"Average acccuracy: {sum(acc_list)/len(acc_list)}")
    print(f"Max accuracy: {max(acc_list)}")
    print(f"Min accuracy: {min(acc_list)}")

