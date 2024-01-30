# solver_icml
anonymous optimizer publication

## Content
This Python code is intended to be able to reproduce the results of ELRA optimizers C2Min + P2Min as part of the actual review process. Code quality could be improved, but the actual state is functional work in progess.

## Authors
Date: 30th Jan. 2024, 
--------
(et al.)

## License
Copyright = none now.

## Structure
Main-File: mnistBenchmark.py, 
Algorithm-Core: lib_grad_solve.py
+ some glue-code-files
+ requirements.txt (pip)

## Environment
Tested under Windows-10 and Ubuntu-22.04 with Python 3.11 and PyTorch-2.1 and (optional) Cuda-12.3 on RTX-4070.

## Usage
# run C2Min, seed=123, epochs=20, net=MNIST+TinyNet
python mnistBenchmark.py -o c2m -n M -b 24 -s 123 -e 20

CAUTION: as we internaly use a dataset cache for small datasets (MNIST*, CIFAR*) it is recommended to delete these files when changing used datasets (-n parameter).
"delete/rm ds_*.pt"

# run P2Min, seed=123, epochs=50, net=CIFAR10+ResNet18
python mnistBenchmark.py -o p2m -n C -b 32 -s 123 -e 50

# run Adam, seed=123, epochs=10, net=CIFAR100+ResNet18
python mnistBenchmark.py -o adam -n 100 -b 256 -s 123 -e 10

# network+dataset (-n ..)
M = MNIST(10), 
F = MNIST-Fashion, 
C = CIFAR10+ResNet18, 
C34 = CIFAR10+ResNet34, 
W = CIFAR10+WideResNet28-10, 
100 = CIFAR100+ResNet18, 
I = TinyImageNet200+ResNet50 (default batchsize=8) [implementation issue with adam + lion]

# optimizer (-o ..)
c2m (bs=24, or 256 for CIRAR-100),
p2m (bs=32, or 48 for CIRAR-100),
adam lion (bs=256),
(for batchsize also see article)

# EOF.
