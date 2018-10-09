# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:13:51 2018

@author: Yinan Li
"""
# -------------------------------------------------------------------------------------------------------------------- #

import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict

# -------------------------------------------------------------------------------------------------------------------- #

def load_image(data_dir):
    '''
    Function for loading and transforming image data to image folders and dataloaders. 
      
    Args:
        data_dir: string, path to datasets
          folder for image data should have 3 sub-folders for train set, validation set and test set respectively
    
    Return:
        imagefolder dict and dataloader dict, each dictionary has 3 keys: train, vaild, test
        
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
                ]),    
        'valid':transforms.Compose([
                transforms.Resize(265), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ]), 
        'test':transforms.Compose([
                transforms.Resize(265), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ])
    }
    
    image_datasets = {
        key: datasets.ImageFolder(data_dir + '/' + key, transform=data_transforms[key])
        for key in list(data_transforms.keys())
    }
    
    dataloaders = {
        key: torch.utils.data.DataLoader(image_datasets[key], batch_size = 32, shuffle = True)
        for key in list(image_datasets.keys())
    }
    
    return image_datasets, dataloaders

# -------------------------------------------------------------------------------------------------------------------- #

def feedforward(input_size, output_size, hidden_layers, drop_p = 0.001):
    ''' 
    Builds a feedforward sequential model with arbitrary hidden layers.
    
    Args: 
        input_size: integer, size of the input
        output_size: integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers
        drop_p: float between 0 and 1, dropout probability
    Returns:
        a sequential model container
    '''
    layer_set =zip([input_size]+hidden_layers[:-1], hidden_layers)
    
    temp = []
    for h1, h2 in layer_set:
        temp += [nn.Linear(h1, h2)]
        temp += [nn.ReLU()]
        temp += [nn.Dropout(drop_p)]
    
    layer_size = len(hidden_layers)
    name1 = ['fc','activation','dropout']*layer_size
    name2 = [str(x) for x in sorted([i+1 for i in list(range(layer_size))]*3)]
    keys  = [x+y for x, y in zip(name1,name2)]
    
    network = list(zip(keys, temp)) + [('fc_last', nn.Linear(hidden_layers[-1], output_size)),
                                       ('output', nn.LogSoftmax(dim=1))]
    
    classifier = nn.Sequential(OrderedDict(network))
    
    
    return classifier

# -------------------------------------------------------------------------------------------------------------------- #

def build_network(arch, output_size, hidden_layers, drop_p):
    ''' 
    Modifies pretrained network, replaces the classifier with new fully-connected layer.
    
    Args:
        arch: string, name of pre-trained model architecture, choice from {'vgg11', 'densenet121'}
        output_size: integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers
        drop_p: float between 0 and 1, dropout probability
            
    Return:
        a pre-trained model with new defined classifier
    
    Raises:
        ValueError: if arch is not equal to 'vgg11' or 'densenet121'  
        ValueError: if the smallest hidden unit is less than output_size 
                    or the largest hidden unit exceeds input_size
        
    '''    
    # Step 1: Load a pre-trained network
    if arch=='vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)
    
    # Step 2: Freeze parameters of pretrained model
    for param in model.parameters():
        param.requires_grad = False
    
    #Step 3: Identify the size of the connection layer between features and classifier
    if list(model.classifier.children()):
        input_size = model.classifier[0].in_features
    else:
        input_size = model.classifier.in_features
    
    #Print input_size, output_size, hidden_layers as references
    print('-'*33)
    print('Number of input features:', input_size)
    print('Number of output features:', output_size)
    print('Size of hidden layers:', hidden_layers)
    print('='*33)
    
    #Step 4: Set rules for hidden_layers & Define a new classifier
    if input_size < max(hidden_layers):        
        raise ValueError('Unexpected size of hidden layers', hidden_layers)
    elif output_size > min(hidden_layers):        
        raise ValueError('Unexpected size of hidden layers', hidden_layers)
    else:
        model.classifier = feedforward(input_size, output_size, hidden_layers, drop_p)
        return model
# -------------------------------------------------------------------------------------------------------------------- #

def validation(model, validloader, criterion, device):
    ''' 
    Function for tracking the loss and accuracy on the validation set.
    
    Args:
        model: pre-trained model with modified architecture
        validloader: dataloader for validation set
        criterion: type of loss function
        device: use GPU if available
            
    Returns:
        average validation loss, float
        average validation accuracy, float

    '''   
    acc, loss = 0,0

    model.to(device)
    for inputs, labels in validloader:        
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
        outputs_ps  = torch.exp(outputs)
        pred_labels =  torch.max(outputs_ps, 1)
        acc += (pred_labels[1] == labels.data).type(torch.FloatTensor).mean().item()
        loss += criterion(outputs, labels).item()
    
    acc = acc/len(validloader)
    loss = loss/len(validloader)
    
    return loss, acc
# -------------------------------------------------------------------------------------------------------------------- #
