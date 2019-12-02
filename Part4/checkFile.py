import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def initFiles(batch_size,valid_size):

    transform = transforms.Compose(
        [transforms.RandomCrop(28,
                               padding=0,
                               pad_if_needed=False),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transformtest = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    # Download/Check Training dataset
    trainData =torchvision.datasets.FashionMNIST(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    print('TrainData Done')
    print('Size Train Data:', len(trainData))

    
    # Split training into train and validation
    indices = torch.randperm(len(trainData))
    train_indices = indices[:len(indices) - valid_size]         # Taining         1 - 59 000 (59 000)
    valid_indices = indices[len(indices) - valid_size:]         # Validation 59 001 - 60 000 ( 1 000)


    # Loading Training Data
    trainloader = torch.utils.data.DataLoader(trainData,
                                             batch_size=batch_size,
                                             sampler=torch.utils.data.SubsetRandomSampler(train_indices),
                                             num_workers=2)
    print('TrainLoader Done')
    print('Size Train:', len(trainloader))

    
    # Loading Validation dataset
    validloader = torch.utils.data.DataLoader(trainData,
                                             batch_size=batch_size,
                                             sampler=torch.utils.data.SubsetRandomSampler(valid_indices),
                                             num_workers=2)
    print('ValidLoader Done')
    print('Size Train:', len(validloader))

    
    # Download/Check Test dataset
    testset = torchvision.datasets.FashionMNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transformtest)
    print('TestSet Done')
    print('Size Train:', len(testset))

    
    # Loading Test dataset
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)
    print('TestLoader Done')
    print('Size Train:', len(testloader))

    
    # Hard defining classes in the Dataset
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    
    # Returing the datasets & class labels
    return trainloader, validloader, testloader, classes