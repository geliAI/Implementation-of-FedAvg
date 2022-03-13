import torch
from torch import nn
from torch.utils.data import DataLoader
import copy

class Server(object):
    def __init__(self,server_model,num_clients = 100):
        self.server_model = server_model
        self.num_clients = num_clients
    
    def update_weights(self,local_weights):

        aggregated_weights = copy.deepcopy(local_weights[0])
        for key in aggregated_weights.keys():
            for i in range(1, len(local_weights)):
                aggregated_weights[key] += local_weights[i][key]
            aggregated_weights[key] = torch.div(aggregated_weights[key], len(local_weights))
            # aggregated_weights[key] = torch.div(aggregated_weights[key], self.num_clients)

        self.server_model.load_state_dict(aggregated_weights)
    
    def inference(self, test_dataset,device):
        """ Returns the test accuracy and loss.
        """

        self.server_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        criterion = nn.NLLLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = self.server_model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss