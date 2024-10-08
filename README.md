# solver_iclr
anonymous optimizer publication

## Content
This Python code is intended to be able to reproduce the results of ELRA optimizer (p2m) as part of the actual review process. Code quality could be improved, but the actual state is functional work in progess.

## Authors
Date: 30th Sep. 2024, 
--------
hidden (et al.)

## License
Copyright = CC BY-SA. This source code is part of a optimizer algorithm review process in year 2024.

## Structure
Main-File: main_elra.py, 
Algorithm-Core: lib_grad_solve.py
+ some glue-code-files
+ requirements.txt (pip)

## Environment
Tested under Windows-10 and Ubuntu-24.04 with Python 3.12 and PyTorch-2.4 and (optional) Cuda-12.4 on RTX-4070.

## Usage
# run ELRA, seed=123, epochs=20, net=MNIST+TinyNet
`python main_elra.py -o elra -n M -b 32 -s 123 -e 20`
Hint: such small batchsize will only work for C2M and P2M! For Adam + Lion use larger values (e.g. 256).

CAUTION: as we internaly use a dataset cache for small datasets (MNIST*, CIFAR*) it is recommended to delete these files when changing used datasets (-n parameter).
`delete/rm ds_*.pt`

# run P2Min, seed=123, epochs=50, net=CIFAR10+ResNet18
`python main_elra.py -o elra -n C -b 32 -s 123 -e 50`

# run Adam, seed=123, epochs=10, net=CIFAR100+ResNet18
`python main_elra.py -o adam -n 100 -b 256 -s 123 -e 10`

# network+dataset (-n ..)
M = MNIST(10), 
F = MNIST-Fashion, 
C = CIFAR10+ResNet18, 
C34 = CIFAR10+ResNet34, 
W = CIFAR10+WideResNet28-10, 
100 = CIFAR100+ResNet18, 
I = TinyImageNet200+ResNet50 (default batchsize=8) [implementation issue with Adam + Lion on TinyImgNet due to GPU-memory]

# optimizer (-o ..)
c2m (bs=24, or 256 for CIRAR-100),
p2m or elra (bs=32, or 48 for CIRAR-100),
adam lion (bs=256),
(for batchsize also see article)

EOF.
