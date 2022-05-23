import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
#from collections import OrderedDict
from PIL import Image
import numpy as np
# Python 3.9.7
#print(torch.__version__)
# torch 1.7.1.post2
#print(torchvision.__version__)
# torchvision 0.8.0a0

# Difference between training with and without the Cutout data augmentation algorithm


class DenseNet3(nn.Module):
    def __init__(self): 
        super().__init__()
        # First convolution
        self.firstconv = nn.Sequential(
                              nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False),
                              nn.BatchNorm2d(64),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
              )

        # Dense Blocks
        self.denseblock1 = self.dense_block(64,32)
        self.denseblock2 = self.dense_block(128,32)
        self.denseblock3 = self.dense_block(128,32) 
        
        # Transition Layers
        self.translayer1 = self.trans_block(192,128)
        self.translayer2 = self.trans_block(256,128)
        
        # Classifier
        self.finalbn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.fc = nn.Linear(1024, 10)  # Linear layer (,num of classes)
             
    def dense_block(self, in_channels, out_channels, num_convs=4):
        layers = []       
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            layers.append(nn.Sequential(
                               nn.BatchNorm2d(in_c),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(in_c, num_convs*out_channels, kernel_size=1, stride=1, bias=False),
                               nn.BatchNorm2d(num_convs*out_channels),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(num_convs*out_channels, out_channels, kernel_size=3, stride=1, padding =1, bias=False)
            ))  

        return torch.nn.Sequential(*layers) 

    def trans_block(self, in_channels, out_channels):
        blk = nn.Sequential(
                   nn.BatchNorm2d(in_channels),            
                   nn.Conv2d(in_channels,out_channels,kernel_size=1, bias=False),
                   nn.AvgPool2d(kernel_size=2, stride = 2,padding =1)
        )
        return blk
    
    def forward(self,x):
        # First convolution
        x = self.firstconv(x)
        
        # Three denseblock
        for i in range(4):
                x = torch.cat((x,self.denseblock1[i](x)), 1)
        x =self.translayer1(x)
        
        for i in range(4):
                x = torch.cat((x,self.denseblock2[i](x)), 1)        
        x = self.translayer2(x)
        
        for i in range(4):
                x = torch.cat((x,self.denseblock3[i](x)), 1)        
        
        #denseblock1 & translayer1
        #conv1 = self.denseblock1[0](x)
        #c1 = torch.cat([x,conv1],1)
        
        #conv2 = self.denseblock1[1](c1)
        #c2 = torch.cat([c1,conv2],1)
        
        #conv3 = self.denseblock1[2](c2)
        #c3 = torch.cat([c2,conv3],1)
           
        #conv4 = self.denseblock1[3](c3)
        #x = torch.cat([c3,conv4],1)
        #x = self.translayer1(x)

        
        # Final
        x = self.avgpool(self.relu(self.finalbn(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)

        return x

def cutout(image,s):
    size = np.random.randint(0,s+1) # uniformly sampled from [0, s]
    c,h,w = image.shape  # Tensor image of size (channels, height, width)
   
    # initial
    y = np.random.randint(h)
    x = np.random.randint(w)
    
    # make sure x,y in feasible interval
    while y-size//2 < 0 or y+size//2 > h:
         y = np.random.randint(h)
        
    while x-size//2 < 0 or x+size//2 > w:
         x = np.random.randint(w)
    
    # np.clip(a, a_min, a_max)
    y1 = np.clip(y-size//2, 0, h)
    y2 = np.clip(y+size//2, 0, h)
    x1 = np.clip(x-size//2, 0, w)
    x2 = np.clip(x+size//2, 0, w)
    
    image[:,y1:y2,x1:x2] = 0 # a square mask
    return image


def dataset_split(data):
    # Split the data into development set and holdout test set with ratio 8:2
    #
    # develop_dataset: development set 
    # test_dataset: holdout test set
    develop_size = int(0.8 * len(data))
    test_size = len(data) - develop_size
    develop_dataset, test_dataset = torch.utils.data.random_split(data, [develop_size, test_size])

    return develop_dataset, test_dataset


def three_cross_valid_cutout(develop_dataset):
    accuracy = []  
    #
    # 3-fold cross-validation scheme, using the development set
    #
    for i in range(3):
        fold_size= len(develop_dataset)// 3
        set_start = i * fold_size # start point
        set_end = (i + 1) * fold_size # end point   
        # connect the other sets
        # consider if the end point exceed total data
        valid_index = list(range(set_start,min(set_end,len(develop_dataset))))
        train_index = list(range(0,set_start))+list(range(min(set_end,len(develop_dataset)),len(develop_dataset)))
        trainloader = Data.DataLoader(Data.Subset(develop_dataset,train_index), batch_size = 100, shuffle= True, num_workers=2)
        validloader = Data.DataLoader(Data.Subset(develop_dataset,valid_index), batch_size = 100, shuffle= True, num_workers=2)
        print('The random split is done.')
        print('The valid dataset has '+ str(len(validloader))+' images.') 
        print('The train dataset has '+ str(len(trainloader))+' images.')
        
        
        ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

        ## train
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
    
            for i, data in enumerate(trainloader, 0):
               # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                #if cutout == True:
                    # train with Cutout data augmentation
                for i in range(len(inputs)):
                    inputs[i] = cutout(inputs[i], 10)  
                #else:
                #    inputs = inputs
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            valid_dataiter = iter(validloader)
            images, labels = valid_dataiter.next()
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            accuracy.append(sum(predicted == labels)/len(labels)) 

    return accuracy


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = DenseNet3()
net = DenseNet3()  
develop_dataset, test_dataset = dataset_split(trainset)
three_cross_valid_cutout(develop_dataset)
print('Training done.')
# save trained model
torch.save(net.state_dict(), 'saved_model_cutout.pt')
print('Model saved.')


def three_cross_valid_nocutout(data):
    accuracy = []  
    #
    # 3-fold cross-validation scheme, using the development set
    #
    for i in range(3):
        fold_size= len(develop_dataset)// 3
        set_start = i * fold_size # start point
        set_end = (i + 1) * fold_size # end point   
        # connect the other sets
        # consider if the end point exceed total data
        valid_index = list(range(set_start,min(set_end,len(develop_dataset))))
        train_index = list(range(0,set_start))+list(range(min(set_end,len(develop_dataset)),len(develop_dataset)))
        trainloader = Data.DataLoader(Data.Subset(develop_dataset,train_index), batch_size = 100, shuffle= True, num_workers=2)
        validloader = Data.DataLoader(Data.Subset(develop_dataset,valid_index), batch_size = 100, shuffle= True, num_workers=2)
        print('The random split is done.')
        print('The valid dataset has '+ str(len(validloader))+' images.') 
        print('The train dataset has '+ str(len(trainloader))+' images.')
        
        
        ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

        ## train
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
    
            for i, data in enumerate(trainloader, 0):
               # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
 
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            valid_dataiter = iter(validloader)
            images, labels = valid_dataiter.next()
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            accuracy.append(sum(predicted == labels)/len(labels)) 

    return accuracy

if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    
model = DenseNet3()
net = DenseNet3()   
three_cross_valid_nocutout(develop_dataset)
print('Training done.')
# save trained model
torch.save(net.state_dict(), 'saved_model_without_cutout.pt')
print('Model saved.')
