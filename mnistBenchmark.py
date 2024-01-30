# mnistBenchmark.py (for ICML 2024)

from time import time as time_time # shrink function table
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from torch.utils.data import TensorDataset, Subset, DataLoader
from torchvision import datasets, transforms
# from torchvision.transforms import v2 # beta-warning (and slow)
from Cos2MinTorchFunctionOptimizer import SelfConstOptimTorch
from NNBenchmark import run_benchmarks
from torch.profiler import profile, record_function, ProfilerActivity
from lion_pytorch import Lion # optional
# from multiprocessing import Process, freeze_support
# from keras_applications.resnet import ResNet50
import argparse

class Net(torch.nn.Module): # 10+10, N= 7960
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, 10)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        z1 = self.fc1(torch.flatten(x, start_dim=1))
        z1 = self.fc2(self.relu(z1))
        return F.log_softmax(z1, dim=1) # log probs are returned
        
class Net16(torch.nn.Module): # 16+16+10
    def __init__(self):
        super(Net16, self).__init__()
        self.fc1 = torch.nn.Linear(784, 16)
        self.acf1 = torch.nn.ReLU()
        self.fc1a = torch.nn.Linear(784, 784)
        self.acf1a = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 16)
        self.acf2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(816, 10) #in_feature:32

    def forward(self, x):
        z1 = self.fc1a(torch.flatten(x, start_dim=1))
        z1a = self.fc1(self.acf1a(z1))
        z2 = self.fc2(self.acf1(z1a))
        #z3 = self.fc3(self.acf2(z2))
        z4 = torch.cat([z1a,z2, z1], 1) # skip-connection
        z3 = self.fc3(self.acf2(z4))
        return F.log_softmax(z3, dim=1) # log probs are returned

# for ResNet
class BasicBlock(nn.Module):
    expansion: int = 1 # hard-coded

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != 1 * planes: # self.expansion
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    1 * planes, # self.expansion
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(1 * planes), # self.expansion
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        return F.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion: int = 4 # ResNet50=4, hard-coded

    def __init__(self, in_planes, planes:int, stride:int=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, 4 * planes, kernel_size=1, bias=False # self.expansion
        )
        self.bn3 = nn.BatchNorm2d(4 * planes) # self.expansion

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != 4 * planes: # self.expansion
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    4 * planes, # self.expansion
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(4 * planes), # self.expansion
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block = BasicBlock, num_blocks = [2, 2, 2, 2], num_classes:int=10, channels:int=3):  #BasicBlock, [2, 2, 2, 2] = R18
        super(ResNet, self).__init__()
        self.in_planes: int = 64 # ResNet50 = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes:int, num_blocks:int, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4) = wrong: https://discuss.pytorch.org/t/1-channel-resnet50-not-working-with-different-input-sizes/81803/6
        out = F.adaptive_avg_pool2d(out, (1, 1)) # bug-fix
        # out = out.view(out.size(0), -1)
        return self.linear(out.view(out.size(0), -1))

##########################################

class wide_basic(nn.Module):
    def __init__(self, in_planes:int, planes:int, dropout_rate:float, stride:int=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        # out += self.shortcut(x)
        return out + self.shortcut(x)

class Wide_ResNet(nn.Module):
    def __init__(self, depth:int, widen_factor:int, dropout_rate:float, num_classes:int):
        super(Wide_ResNet, self).__init__()
        self.in_planes: int = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n: int = int( (depth-4) / 6)
        k: int = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k)) # 28x10
        nStages3: int = 64*k # nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True) # nStages[0], conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, 16*k, n, dropout_rate, stride=1) # nStages[1]
        self.layer2 = self._wide_layer(wide_basic, 32*k, n, dropout_rate, stride=2) # nStages[2]
        self.layer3 = self._wide_layer(wide_basic, nStages3, n, dropout_rate, stride=2) # nStages[3]
        self.bn1 = nn.BatchNorm2d(nStages3, momentum=0.9)
        self.linear = nn.Linear(nStages3, num_classes)

    def _wide_layer(self, block, planes:int, num_blocks:int, dropout_rate:float, stride:int):
        strides = [stride] + [1]*((num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # assert(x is not None) # first is hit
        # out = self.conv1(x)
        out = self.layer1( self.conv1(x) )
        # out = self.layer2(out)
        out = self.layer3( self.layer2(out) )
        out = F.relu( self.bn1(out) )
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return self.linear(out)

##########################################

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2]) # cl=10

def ResNet18_100(): # ParamCount = 11220132, loss0 = 4.728
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet34_100():
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=100)

def ResNet50():
    # return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200) # n=23910152

def WRN28():
    return Wide_ResNet(28, 10, 0.3, 10) # n=36489290
    # return WideResNet(d=28, k=10, n_classes=10, input_features=3, output_features=16, strides=[1, 1, 2, 2]) # TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not tuple
    # WideResNet(28, 10, widen_factor=1, dropRate=0.0)

#def ResNet101():    return ResNet(Bottleneck, [3, 4, 23, 3])
#def ResNet152():    return ResNet(Bottleneck, [3, 8, 36, 3])

class FashionCNN(nn.Module): # from D.
    def __init__(self):
        super().__init__()
        self.conv_relu_pool_stack = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2), # 14x14

            # Layer 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2), # 7x7

            # Layer 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(49*256, 10),
        )

    def forward(self, x):
        x = self.conv_relu_pool_stack(x)
        return self.classifier(x)

class FashionCNN2(nn.Module):
    
    def __init__(self):
        super(FashionCNN2, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout1d(0.25) # UserWarning: Dropout2d
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out = out.view(out.size(0), -1)
        out = self.drop(self.fc1(out))
        out = self.fc3(self.fc2(out))
        return out # test
        return F.log_softmax(out, dim=1)

######################################################################

alphas = [0.001] #10**np.array(range(1, -3, -1), dtype = float)          # learning rates to test for each optimizer, CAUTION: small learning rates can take very long for some optimizers
loss_func = None #torch.nn.NLLLoss() # CIFAR=nn.CrossEntropyLoss()
optimizer = {
    #"VGD"               :   (torch.optim.SGD,       [[alpha]           for alpha in alphas]),
    #"RMSprop"           :   (torch.optim.RMSprop,   [[alpha]           for alpha in alphas]),
    #"Adagrad"           :   (torch.optim.Adagrad,   [[alpha]           for alpha in alphas]),
    #"Adadelta"          :   (torch.optim.Adadelta,  [[alpha]           for alpha in [1.0]]),
    #"Adam"              :   (torch.optim.Adam,      [[alpha]           for alpha in alphas]),
    #"Adamax"            :   (torch.optim.Adamax,    [[alpha]           for alpha in alphas]),
    #"AdamW"             :   (torch.optim.AdamW,     [[alpha]           for alpha in alphas])
    "c2min_vanilla"     :   (SelfConstOptimTorch,   [(alpha, SelfConstOptimTorch.Mode.c2min, loss_func)            for alpha in alphas]),
    #"c2min_check"       :   (SelfConstOptimTorch,   [(alpha, SelfConstOptimTorch.Mode.c2min_check, loss_func)      for alpha in alphas]),    
    #"p2min_function"    :   (SelfConstOptimTorch,   [(alpha, SelfConstOptimTorch.Mode.p2min, loss_func)   for alpha in alphas])
}

def Loader2(batch_size:int = 8, valid_size:float = 0.125):
    "unused"
    num_workers:int = 4
    train_data = datasets.ImageFolder('/train/', transform=data_transform_train)
    valid_data = datasets.ImageFolder('/train/', transform=data_transform_val)

    # obtain training indices that will be used for validation
    num_train:int = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split:int = int(np.floor(valid_size * num_train))
    # train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])

    # prepare data loaders 
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    
    # train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    # val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    return train_loader, valid_loader

def dataset_split2(ds, cut:list):
    assert(len(cut) == 2), "empty cut list"
    assert(ds is not None), "type=None"
    assert(len(ds) > 1), "empty dataset"
    if (sum(cut) != len(ds)):
        if (cut[0] == 0): cut[0] = len(ds) - sum(cut)
        if (cut[1] == 0): cut[1] = len(ds) - sum(cut)
    return torch.utils.data.random_split(ds, cut)

def ArgParsing():
    parser = argparse.ArgumentParser(prog='mnistBenchmark.py')
    parser.add_argument('--hilfe', help='help: development state (--seed=3)')
    parser.add_argument('-s', '--seed', type=int) # srand
    parser.add_argument('-e', '--epoch', type=int) # max
    parser.add_argument('-b', '--bs', type=int) # batch_size
    parser.add_argument('-i', '--inits', type=int)
    parser.add_argument('-a', '--alpha', type=float) # alpha0
    parser.add_argument('-t', '--target', type=float)
    parser.add_argument('-o', '--solver', type=str) # optimizer
    parser.add_argument('-d', '--device', type=int) # -1=cpu,0=gpu:0,..
    parser.add_argument('-n', '--net', type=str) # MFCI: MNIST-16, CIFAR+Resnet18, ImageNet+Resnet50
    return parser.parse_args()

def ChooseOpt(solver: str, alphas: list[float]):
    if (solver is None) or (len(solver) < 3) or (len(solver) > 8):
        print("--solver: 2<LEN<8", solver)
        exit()
        return None

    assert(len(alphas) > 0), "list[LR] empty"
    # assert(loss_func is not None)
    solver = solver.lower()

    if (solver == "c2m"):
        return {
            "c2min_vanilla" : (SelfConstOptimTorch,  [(alpha, SelfConstOptimTorch.Mode.c2min, loss_func)  for alpha in alphas])
        }
    if (solver == "p2m"):
        return {
            "p2min_function" : (SelfConstOptimTorch,  [(alpha, SelfConstOptimTorch.Mode.p2min, loss_func)  for alpha in alphas])
        }
    if (solver == "c2c"):
        return {
            "c2min_check" : (SelfConstOptimTorch,  [(alpha, SelfConstOptimTorch.Mode.c2min_check, loss_func)  for alpha in alphas])
        }
    if (solver == "adam"):
        return {
            "Adam" : (torch.optim.Adam, [[alpha]  for alpha in alphas])
        }
    if (solver == "lion"):
        if (alphas[0] >= 0.001): alphas = [1e-4] # Lion (1e-4) and Adam (1e-3), check
        return {
            "Lion" : (Lion, [[alpha]  for alpha in alphas])
        }

    assert(0), "unknown solver-param"
    return None

def GetDevice(did: int) -> int:
    if (not cuda.is_available()):
        return torch.device("cpu") # default=cpu
    if (did is None):
        return torch.device("cuda")
    if (did < 0):
        return torch.device("cpu")
    return torch.device("cuda", int(did))

def main() -> None:
    global optimizer, alphas
    my_seed: int = -4 # reproducible (93, 45, ..)
    initializations: int = 1 #5
    batchings: int = 1 #5
    max_epochs: int = 40 #10
    target_loss: float = 0.0001 #0.33
    batch_sizes = [256] # [256,512,1024] #,2048,4096] # fastet

    args = ArgParsing()
    if (args.seed is not None): my_seed = int(args.seed)
    if (args.epoch is not None): max_epochs = int(args.epoch)
    if (args.bs is not None): batch_sizes = [ int(args.bs) ]
    if (args.inits is not None): initializations = int(args.inits)
    if (args.alpha is not None): alphas = [ abs(args.alpha) ]; print("alpha:TODO", args.alpha)
    if (args.target is not None): target_loss = args.target
    if (args.solver is not None):
        optimizer = ChooseOpt(args.solver, alphas)
    dnet: str = "M" if (args.net is None) else args.net.upper() # "MCI", TODO

    start_overall_time: float = time_time()
    torch.manual_seed(my_seed)

    # torch.set_default_dtype(torch.float16) # default off (32bit)
    device = GetDevice(args.device)

    # In Console: nvidia-smi
    print("CUDA:", cuda.is_available(), ", dev=", device, "x", cuda.device_count(), ", dtype=", str(torch.get_default_dtype()).replace('torch.',''), ", seed=", my_seed)

    # ImgNet: 1000 object classes, 1,281,167 training images, 50,000 validation images and 100,000 test images, 469x387 usually cropped to 256x256 or 224x224 pixels
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) # default CIFAR test
        
    if ("M" == dnet[0]): # MNIST: "M" + "M16" = 10 class, 60k+10k x 28x28.
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), # old+faster(40%)
            # v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            # v2.Normalize((0.1307,), (0.3081,))
        ])
        run_benchmarks(
            datasets.MNIST('../data', train=True, download=True, transform=transform),
            datasets.MNIST('../data', train=False, transform=transform),
            model_class = Net if (dnet != "M16") else Net16,
            model_hyperparams=(),
            optimizers=optimizer,  runs=initializations,
            batch_sizes=batch_sizes, loss_func=torch.nn.NLLLoss(),
            target_loss=target_loss, max_epochs=max_epochs, different_batch_sets = batchings,
            task_name="mnist", device=device, seed0=my_seed)

    if ("F" == dnet[0]): # FashionMNIST, 70000 28x28 grayscale images 10 cat.
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
        ])
        run_benchmarks(
            datasets.FashionMNIST('../data', train=True, download=True, transform=transform),
            datasets.FashionMNIST('../data', train=False, transform=transform),
            model_class = FashionCNN if (dnet != "F16") else Net16,
            model_hyperparams=(),
            optimizers=optimizer,  runs=initializations,
            batch_sizes=batch_sizes, loss_func=torch.nn.CrossEntropyLoss(),
            target_loss=target_loss, max_epochs=max_epochs, different_batch_sets = batchings,
            task_name="mnistF", device=device, seed0=my_seed)

    if ("C" == dnet[0]): # CIFAR: "C" = 10 class, 50k+10k, 3x32x32
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'), # why?
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        run_benchmarks(
            datasets.CIFAR10('../data', train=True, download=True, transform=tf_train),
            datasets.CIFAR10('../data', train=False, transform=tf_test),
            model_class = ResNet18 if (len(dnet)==1) else ResNet34,
            model_hyperparams=(),
            optimizers=optimizer,  runs=initializations,
            batch_sizes=batch_sizes, loss_func=nn.CrossEntropyLoss(),
            target_loss=target_loss, max_epochs=max_epochs, different_batch_sets = batchings,
            task_name="cifar", device=device, seed0=my_seed)

    if (0 == dnet.find("100")): # CIFAR: "C" = 100 class, fast(15s/ep), CIFAR-100(2019=91.7%..96),bs=256..512++
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        run_benchmarks(
            datasets.CIFAR100('../data', train=True, download=True, transform=tf_train),
            datasets.CIFAR100('../data', train=False, transform=tf_test),
            model_class = ResNet18_100 if (len(dnet)==3) else ResNet34_100,
            model_hyperparams=(),
            optimizers=optimizer,  runs=initializations,
            batch_sizes=batch_sizes, loss_func=nn.CrossEntropyLoss(),
            target_loss=target_loss, max_epochs=max_epochs, different_batch_sets = batchings,
            task_name="cifar100", device=device, seed0=my_seed)

    if ("W" == dnet[0]): # Wide_ResNet28 + CIFAR10 = 10 class, (3,32,32), 2min/ep (runs)
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=4)
        run_benchmarks(
            datasets.CIFAR10('../data', train=True, download=True, transform=tf_train),
            datasets.CIFAR10('../data', train=False, transform=tf_test),
            model_class = WRN28, # n=36489290=36mio, ac:88.5%/10ep
            model_hyperparams=(),
            optimizers=optimizer,  runs=initializations,
            batch_sizes=batch_sizes, loss_func=nn.CrossEntropyLoss(),
            target_loss=target_loss, max_epochs=max_epochs, different_batch_sets = batchings,
            task_name="cifarW28", device=device, seed0=my_seed)

    if ("I" == dnet[0]): # ImageNet+Resnet50: "I", classes=200, image_size=64x64x3,
        gpu_gb: int = -1 if (not cuda.is_available()) else cuda.get_device_properties(0).total_memory >> 30
        # 500x200 x 64x64x3= 100000x12288 = 1229 mio Bytes = 1172 MB
        # 500x200 x 224x224x3 = 100000x150528 x 32bit => 294 MB / 512 batch = 18.4 GB / 32K batch
        # on V100 GPU, time to train on CIFAR-100 + Tiny-ImageNet = about 30 min + 200 min per round
        print("Warning: TinyImagenet, RAM = 3..5 GB (use small batches 8..32) !!", gpu_gb)
        tf = transforms.Compose([ # 64x4=256, later crop 16+224+16
            transforms.Resize(256), transforms.CenterCrop(224),
            # transforms.CenterCrop(224+8), transforms.RandomCrop(224), # disadvantage!
            transforms.RandomHorizontalFlip(p=0.5), # new (anti-overfit)
            transforms.ToTensor(), # old
            # v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # new/slow(!10x!)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tft = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            # v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # new/slow(!10x!)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None) # dim=25557032=25M
        # tiny_imagenet = load_dataset('../data/tiny-imagenet-200', split='train')
        # TODO: "archive ILSVRC2012_img_train.tar is not present"
        dataset = datasets.ImageFolder('../data/tiny-imagenet-200/train', transform=tf) # 100K
        # testset = datasets.ImageFolder('../data/tiny-imagenet-200/test', transform=tft) # 5+5K (test+val) == CAUTION: 
        dataset, testset = torch.utils.data.random_split(dataset, [94000, 6000]) # 
        if (gpu_gb < 11): # small GPU RAM
            print("DS-CUT:", len(dataset), len(testset))
            dataset, _ = dataset_split2(dataset, [2000, 0])
            testset, _ = dataset_split2(testset, [1000, 0])
            _ = None
        # parts = [32000, 8000, 60000] if (gpu_gb < 11) else [86000, 14000, 0]
        # parts = [120, 80, 99800] # debug
        # train_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, parts) # [32000, 8000, 60000]
        # dataset, _ = None, None # release GBs of RAM
        if (batch_sizes[0] > 128): batch_sizes = [ 8 ] # default for ResNet50 (bs=32 works with >16GB)
        print("GPU=%d+1GB, bs=%d, ep=%d, %d + %d" % (gpu_gb, batch_sizes[0], max_epochs, len(dataset), len(testset)))

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

        run_benchmarks(
            dataset, testset,
            model_class = ResNet50,
            model_hyperparams=(),
            optimizers=optimizer,  runs=initializations,
            batch_sizes=batch_sizes, loss_func=nn.CrossEntropyLoss(),
            target_loss=target_loss, max_epochs=max_epochs, different_batch_sets = batchings,
            task_name="imgnet", device=device, seed0=my_seed)

    # prof.export_chrome_trace("trace.json")

    runtime: float = time_time() - start_overall_time
    print("Runtime: %.6f seconds" % (runtime))
    return
# main.

if __name__ == "__main__":
    main()
    # freeze_support() # DataLoader (worker-threads) fail on Windows10 + Cuda12

#EoF.
