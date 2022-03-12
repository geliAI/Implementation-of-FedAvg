import argparse

class ArgsInit(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Implementation of the FederatedAveraging Algorithm')

        # model arguments
        parser.add_argument('--arch', type=str, default='mlp', help='architecture used. cnn or mlp')

        # federated arguments (Notation for the arguments followed from paper)
        parser.add_argument('--epochs', type=int, default=10,
                            help="number of rounds of global training")
        parser.add_argument('--num_clients', type=int, default=100,
                            help="number of clients: K")
        parser.add_argument('--frac', type=float, default=0.1,
                            help='the fraction of clients selected: C')
        parser.add_argument('--local_batch_size', type=int, default=10,
                            help="local batch size: B")
        parser.add_argument('--local_epoch_num', type=int, default=10,
                            help="the number of epochs of local training: E")
        parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate')
        parser.add_argument('--momentum', type=float, default=0.5,
                            help='SGD momentum (default: 0.5)')

        # data arguments
        parser.add_argument('--dataset', type=str, default='mnist', help="dataset to be used")
        parser.add_argument('--num_class', type=int, default=10, help="number of data classes")
        parser.add_argument('--iid', type=int, default=1,
                            help='Default set to IID. Set to 0 for non-IID.')

        # training arguments
        parser.add_argument('--gpu', default=None, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")
        parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                            of optimizer")
        parser.add_argument('--stopping_rounds', type=int, default=10,
                            help='rounds of early stopping')
        parser.add_argument('--verbose', type=int, default=1, help='verbose')
        parser.add_argument('--seed', type=int, default=1, help='random seed')


        self.args = parser.parse_args()

    def get_args(self):
        return self.args