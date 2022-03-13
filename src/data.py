import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def mnist_iid(dataset, num_clients = 100):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: PyTorch Dataset
    :param num_clients: number of client models
    :return: A dict mapping client id to image index, where the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    examples_per_client = int(len(dataset)/num_clients)
    client_2_img, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        client_2_img[i] = set(np.random.choice(all_idxs, examples_per_client,
                                             replace=False))
        all_idxs = list(set(all_idxs) - client_2_img[i])
    return client_2_img

def mnist_noniid(dataset, num_clients = 100,shards_per_client = 2):
    """
    Sample non-I.I.D client data from MNIST dataset.
    :param dataset:
    :param num_users:
    :return:
    """
    #num_shards =  200, shard_size = 300
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards = num_clients * shards_per_client
    shard_size = len(dataset) // num_shards

    assert shard_size*shards_per_client*num_clients == len(dataset)
    
    idx_shard = [i for i in range(num_shards)]
    client_2_img = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards*shard_size)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards per client
    np.random.shuffle(idx_shard)
    for i in range(num_clients):
        shards_selected = idx_shard[shards_per_client*i:shards_per_client*i+shards_per_client]
        for shard in shards_selected:
            client_2_img[i] = np.concatenate(
                (client_2_img[i], idxs[shard*shard_size:(shard+1)*shard_size]), axis=0)
    return client_2_img

    # for i in range(num_clients):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         client_2_img[i] = np.concatenate(
    #             (client_2_img[i], idxs[rand*shard_size:(rand+1)*shard_size]), axis=0)
    # return client_2_img

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def prepare_dataset(data_dir,dataset_name='mnist',iid=1,num_clients=100):
    """ Returns train and test datasets and a client_2_img where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset_name == 'mnist':
    
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if iid:
            # Sample IID user data from Mnist
            client_2_img = mnist_iid(train_dataset, num_clients = num_clients)
        else:
            # Sample Non-IID user data from Mnist
            # Chose euqal splits for every user
            client_2_img = mnist_noniid(train_dataset, num_clients = num_clients,shards_per_client = 2)
    else:
        exit('Not implemented.')

    return train_dataset, test_dataset, client_2_img

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num_clients = 100,shards_per_client = 2)