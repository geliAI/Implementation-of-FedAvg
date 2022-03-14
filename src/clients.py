import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from data import DatasetSplit


class LocalClient(object):
    def __init__(self, client_id,dataset,image_list,local_bs,lr=0.1,local_epoch_num=10,device='cpu'):
        self.client_id = client_id
        self.device = device
        self.lr = lr
        self.local_ep = local_epoch_num
        self.local_batch_size = local_bs
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(image_list))

        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.local_batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def ClientUpdate(self, model, global_round,verbose=0):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                        momentum=0.5)

        for local_epoch in range(self.local_ep):
            # print('Start Local Training on Client : {client_id}\n'.format(client_id=self.client_id))

            batch_loss = []
            correct = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                _, pred_labels = torch.max(log_probs, 1)
                pred_labels = pred_labels.view(-1)

                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # if batch_idx % 20 == 0:
                #     print('| Global Round : {} | Local Epoch : {} on Client : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round+1, iter+1, self.client_id,
                #         batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            
                
            current_epoch_loss = sum(batch_loss)/len(batch_loss)
            current_epoch_acc = correct/total
            if verbose == 1:
                print('| Global Round : {} | Local Epoch : {} on Client : {:2d} |\tAccuracy: {:.3f}|\tLoss: {:.6f}'.format(global_round+1, local_epoch+1, self.client_id, 
                        current_epoch_acc,current_epoch_loss))

            epoch_loss.append(current_epoch_loss)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)