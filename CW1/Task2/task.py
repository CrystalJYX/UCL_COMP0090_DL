import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
#from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
# Python 3.9.7
#print(torch.__version__)
# torch 1.7.1.post2
#print(torchvision.__version__)
# torchvision 0.8.0a0

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
    
# print the network architecture  
if __name__ == '__main__':
    net = DenseNet3()
    print(net)
    
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

if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 100
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # example images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    
    
    # cutout images
    cutout_img = torch.zeros((16,3,32,32)) # Tensor sizes: [3, 32, 32]
    # 16 images
    for i in range(16):
        cutout_img[i] = cutout(images[i], 10)


    cutout_im = Image.fromarray((torch.cat(cutout_img.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    cutout_im.save("cutout.png")
    print('cutout.png saved.')
    
if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 100
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



## DenseNet3
net = DenseNet3()


## loss and optimiser
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)


## train
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # train with Cutout data augmentation
        for i in range(len(inputs)):
            inputs[i] = cutout(inputs[i], 10) 
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

print('Training done.')

# save trained model
torch.save(net.state_dict(), 'saved_model.pt')
print('Model saved.')


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 36

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## load the trained model
    model = DenseNet3()
    model.load_state_dict(torch.load('saved_model.pt'))

    ## inference
    images, labels = dataiter.next()
    print('Ground-truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(36)))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

    # save to images
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("result.png")
    print('result.png saved.')


accuracy = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        accuracy.append(sum(predicted == labels)/len(labels))        
