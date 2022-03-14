import torch
import numpy as np
from args import ArgsInit
from models import MLP, CNN
from clients import LocalClient
from server import Server
from data import prepare_dataset
import copy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    args = ArgsInit().get_args()
    
    arch_selected = args.arch
    global_epoch_num = args.G

    num_clients = args.K
    client_fraction = args.C


    local_epoch_num = args.E
    local_batch_size = args.B
    lr = args.lr
    data_dir = args.data_dir

    # Model Construction with default parameters used in the paper.
    # A dropout layer is used between fc layers for regularization purposes.
    if arch_selected == 'mlp':
        # Convolutional neural netork
        model = MLP(dim_in=784, dim_hidden=200, num_class=10,p_dropout=0.5)
        server = Server(model)
        server.server_model.to(device)

    elif arch_selected == 'cnn':
        # Multi-layer preceptron
        model = CNN(dim_hidden = 512, num_class=10, p_dropout=0.5)
        server = Server(model)
        server.server_model.to(device)

    else:
        exit('Model not implemented.')
    
    # Data patrition preparation
    train_dataset, test_dataset, client_2_img_dict = prepare_dataset(data_dir,dataset_name='mnist',iid=1,num_clients=num_clients)

    # Start global round
    for epoch in range(global_epoch_num):
        local_weights = []
        local_losses = []

        # Randomly select clients to be used:
        # calculate the number of client to be sampled each time : m. 
        m = max(int(client_fraction * num_clients), 1) 
        selected_clients_idx = np.random.choice(range(num_clients), m, replace=False)
        print(f'Start Gloabl Training Round : {epoch+1}  Clients : {selected_clients_idx}\n'.format(epoch=epoch + 1))

        server.server_model.train()
        for client_id in selected_clients_idx:
            
            client = LocalClient(client_id,train_dataset, client_2_img_dict[client_id],local_bs=local_batch_size,lr=lr,
                                        local_epoch_num=local_epoch_num, device=device)

            w, loss = client.ClientUpdate(model=copy.deepcopy(server.server_model), global_round=epoch)

            local_weights.append(w)
            local_losses.append(loss)
        
        avg_train_losses = np.mean(np.array(local_losses))
        # Update server model by taking the average of local weights
        server.update_weights(local_weights)

        test_acc, test_loss = server.inference(test_dataset,device)
        print(f' \n Results after {epoch+1} rounds of global training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Avg Train Loss: {:.6f}".format(avg_train_losses))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    
    test_acc, test_loss = server.inference(test_dataset,device)

    print(f' \n Results after {global_epoch_num} rounds of global training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

if __name__ == "__main__":
    main()