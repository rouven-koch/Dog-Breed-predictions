#-------------------------------------------------------------------------------------------------
# Import libraries
#-------------------------------------------------------------------------------------------------
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline  
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import os
from torchvision import datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True  # fixes problem of truncated images

#-------------------------------------------------------------------------------------------------
# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))
                           
#-------------------------------------------------------------------------------------------------
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
    
# Different classifier
face_cascade_2 = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Returns "True" if face is detected in image stored at img_path
def face_detector_2(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_2.detectMultiScale(gray)  
    return len(faces) > 0
    
#-------------------------------------------------------------------------------------------------
# Define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
    
from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):

    # Open image from path
    image = Image.open(img_path)  
    
    # Transform loaded image
    transform_images = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    
    image = transform_images(image)
    image = image.unsqueeze(0)   # required format for VGG-16
    #image = image.cpu()
    #VGG16 = VGG16.cpu()
    output = VGG16(image) 
    index = output.data.numpy().argmax()
        
    return index
    
# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    if ( VGG16_predict(img_path) >= 151 ) and ( VGG16_predict(img_path) <= 268 ):
        return True
    else:
        return False
        
#-------------------------------------------------------------------------------------------------        
# define training and validation of the model    
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    train_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        #for batch_idx, (data, target) in enumerate(loaders[0]):
        for data, target in train_loader:

            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()           
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

            
        ######################    
        # validate the model #
        ######################
        
        with torch.no_grad():
            model.eval()
            for data, target in valid_loader:
            #for batch_idx, (data, target) in enumerate(loaders[2]):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)

        # average loss calculation
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss,valid_loss))
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss    

    return model

#-------------------------------------------------------------------------------------------------    
 # define model testing
 def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    #for batch_idx, (data, target) in enumerate(loaders[1]):
    for data, target in valid_loader:

        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        #test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        test_loss = loss.item()*data.size(0)
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
        
        
#-------------------------------------------------------------------------------------------------
# Create new model using transfer learning and the VGG-16 model
batch_size = 10
num_workers = 0

data_dir  = '/data/dog_images/'
train_dir = os.path.join(data_dir, 'train/')
test_dir  = os.path.join(data_dir, 'test/')
valid_dir = os.path.join(data_dir, 'valid/')

# Data transformations (maybe different in training and test / validation data)
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data  = datasets.ImageFolder(test_dir,  transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers)


import torchvision.models as models
import torch.nn as nn

# define VGG16 model
model_transfer = models.vgg16(pretrained=True)

# Freeze feature layers
for param in model_transfer.features.parameters():
    param.requires_grad = False

# Modify classifier for 133 Dog Breeds
input_dim = model_transfer.classifier[6].in_features
model_transfer.classifier[6] = nn.Linear(input_dim, 133)    
    
# check if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    model_transfer = model_transfer.cuda()
    
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.0005)

# train the model
n_epochs = 50
loaders_transfer = [train_loader, test_loader, valid_loader]
model_transfer = train(n_epochs, loaders_transfer, model_transfer,
                       optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer_new.pt')
model_transfer.load_state_dict(torch.load('model_transfer.pt'))

test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)

#-------------------------------------------------------------------------------------------------
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    image = Image.open(img_path)  
    
    # Transform loaded image
    transform_images = transforms.Compose([transforms.CenterCrop(244),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    
    image  = transform_images(image)
    image  = image.unsqueeze(0)   
    image  = image.cuda()
    output = model_transfer(image) 
    output = output.data.cpu()
    index  = output.data.numpy().argmax()
    
    return class_names[index]
    
#-------------------------------------------------------------------------------------------------
def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    if face_detector(img_path) == True:
        result = predict_breed_transfer(img_path)
        return('The human on this picture looks like a "' + str(result) + '". ')
    if dog_detector(img_path) == True:
        result = predict_breed_transfer(img_path)
        return('The dogs breed is "' + str(result) + '". ' )
    else:
        print('Error: Sorry, no human or dog detected in this picture!')
        
 
