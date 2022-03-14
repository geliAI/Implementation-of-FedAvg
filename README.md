# Implementation of FedAvg

## Requirements
- Python3
- PyTorch
- Torchvision
To install all requirements, run:

`pip3 install -r requirements.txt`

## How to run an experiment
An example on running experiments

`python3 src/main.py --arch=cnn --G=200 --B=50 --E=5 --iid=1 `

## Arguments
### Key Arguments
- --arch: architecture used, 'mlp' or 'cnn'
- --G: global round number
- --B: local batch size
- --E: local epoch number
- --iid: 1: iid data patrition; 0: non0-iid patrition
### Other Arguments
- --C: fraction of clients used, default : 0.1
- --lr: leaning rate, default is 0.01
- --verbose, control verbose details. default: 0.