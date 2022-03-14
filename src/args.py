import argparse

class ArgsInit(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Implementation of the FederatedAveraging Algorithm')

        # model arguments
        parser.add_argument('--arch', type=str, default='mlp', help='architecture used. cnn or mlp')

        # federated arguments (Notation for the arguments followed from paper)
        parser.add_argument('--G', type=int, default=10,
                            help="global_epoch_num: G")
        parser.add_argument('--K', type=int, default=100,
                            help="number of clients: K")
        parser.add_argument('--C', type=float, default=0.1,
                            help='the fraction of clients selected: C')
        parser.add_argument('--B', type=int, default=10,
                            help="local batch size: B")
        parser.add_argument('--E', type=int, default=1,
                            help="the number of epochs of local training: E")
        parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate')
      
        # data arguments
        parser.add_argument('--dataset', type=str, default='mnist', help="dataset to be used")
        parser.add_argument('--data_dir', type=str, default='./data/mnist', help="dataset to be used")

        parser.add_argument('--num_class', type=int, default=10, help="number of data classes")
        parser.add_argument('--iid', type=int, default=1,
                            help='Default set to IID. Set to 0 for non-IID.')

        # training argument
        parser.add_argument('--verbose', type=int, default=0, help='verbose')


        self.args = parser.parse_args()

    def get_args(self):
        return self.args