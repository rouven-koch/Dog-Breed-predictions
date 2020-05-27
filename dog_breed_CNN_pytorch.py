#----------------------------------------------------------------------------------------------------------------------------
# Import libraries
#----------------------------------------------------------------------------------------------------------------------------
import os
import torch
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # fixes problem of truncated images

#----------------------------------------------------------------------------------------------------------------------------
# Specify appropriate transforms, and batch_sizes
data_dir  = '/data/dog_images/'
train_dir = os.path.join(data_dir, 'train/')
test_dir  = os.path.join(data_dir, 'test/')
valid_dir = os.path.join(data_dir, 'valid/')

# Data transformations 
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data  = datasets.ImageFolder(test_dir,  transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

print( ' Number of images in ..')
print(' ')
print(' Training data: ', len(train_data))
print(' Test data: ', len(test_data))
print(' Valid data: ', len(test_data))

# Dataloaders
batch_size  = 20
num_workers = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, num_workers=num_workers, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Find number of classes
#labels_test = []
#for i in range(len(train_data)):
#    labels_test.append(train_data[i][1])
#print(max(labels_test))  # from 0 to 132 classes!
loaders_scratch = [train_loader, test_loader, valid_loader]

#----------------------------------------------------------------------------------------------------------------------------
# Define CNN model from scratch
#----------------------------------------------------------------------------------------------------------------------------
# check if CUDA is available
use_cuda = torch.cuda.is_available()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)    
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Linear layers (Classifier)
        self.fc1 = nn.Linear(14*14*256, 1024)
        self.fc2 = nn.Linear(1024, 133)
      
        # Dropout
        self.dropout = nn.Dropout(0.275)
    
    def forward(self, x):
        # Define forward behavior
        x = self.pool(F.relu(self.conv1(x)))  # 112
        x = self.pool(F.relu(self.conv2(x)))  #  56
        x = self.pool(F.relu(self.conv3(x)))  #  28
        x = self.pool(F.relu(self.conv4(x)))  #  14

        x = x.view(-1, 14*14*256)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # no sigmoid because of choosen criterion
        
        return x
        
# instantiate the CNN
model_scratch = Net()
print(model_scratch)

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()

# Define criterion and optimizer
import torch.optim as optim
criterion_scratch = nn.CrossEntropyLoss()
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.025)

#----------------------------------------------------------------------------------------------------------------------------
# Define testing and training of the model
#----------------------------------------------------------------------------------------------------------------------------
# Training
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    # train_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders[0]):

            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders[2]):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        ## Save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss    
        
    return model
    
# train the model
model_scratch = train(25, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
#model_scratch.load_state_dict(torch.load('model_scratch.pt'))

#----------------------------------------------------------------------------------------------------------------------------
# Testing

model_scratch.load_state_dict(torch.load('model_scratch.pt')) 

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders[1]):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
    
    
    
