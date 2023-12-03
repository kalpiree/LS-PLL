# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import time
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from torchvision import models, datasets, transforms

import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set a seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the ResNet-18 model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # Corrected super() call
        #self.resnet18 = resnet18(pretrained=False, num_classes=10)
        self.resnet18 = models.resnet18(pretrained=True)
        # Modify the final fully connected layer for 10 classes
        self.resnet18.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet18(x)


# Wrap the model with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(Net())
else:
    model = Net()

# Move the model to the GPU
model = model.to(device)
# Initialize variables to store training and testing predictions
# train_predictions = []
# test_predictions = []

# Load CIFAR-10 dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train,)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

def train_model(model, train_loader, criterion, optimizer, num_epochs=150):
    model.train()


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, ground_truth_label) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, ground_truth_label = inputs.to(device), ground_truth_label.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, ground_truth_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += ground_truth_label.size(0)
            correct += predicted.eq(ground_truth_label).sum().item()

        print(f"Train Loss: {train_loss / (batch_idx + 1):.3f}, Train Accuracy: {100. * correct / total:.2f}%")

    return model  # Return the trained model

# Create DataLoaders for training and testing
train_loader = DataLoader(trainset, batch_size=256, shuffle=True,num_workers=10)
test_loader = DataLoader(testset, batch_size=100, shuffle=False,num_workers=10)
# Define your loss criterion and optimizer
#model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Train the model
model_trained = train_model(model, train_loader, criterion, optimizer, num_epochs=150)

train_loader = DataLoader(trainset, batch_size=256, shuffle=False)
# model_trained.eval()
train_features = []
train_predictions = []
train_targets = []

with torch.no_grad():
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)
        outputs = model_trained(inputs)
        train_predictions.append(outputs.cpu())
        # print(inputs.shape)
        train_features.append(inputs.cpu().numpy())
        train_targets.append(target.cpu().numpy())

train_features_np = np.concatenate(train_features, axis=0)
train_label_set = np.concatenate(train_predictions, axis=0)
train_ground_truth = np.concatenate(train_targets, axis=0)

np.save('./train_features.npy', train_features_np)
np.save('./train_label_set.npy', train_label_set)
np.save('./train_ground_truth.npy', train_ground_truth)

import numpy as np

# Step 1
target_set = np.load('train_label_set.npy')
true_label = np.load('train_ground_truth.npy')

print("target_set_train[0]->",target_set[0])
print("true_label_train[0]->",true_label[0])


# Move data to the GPU
#target_set = torch.tensor(target_set).to(device)
#true_label = torch.tensor(true_label).to(device)

# Step 2
label_set = np.empty((50000, 6))

ground_truth_val = np.empty((50000, 1))
# Iterate
for i in range(50000):
    row = target_set[i]
    index_to_remove = true_label[i]
    ground_truth_val[i] = row[index_to_remove]
    row = np.delete(row, index_to_remove)
    sorted_row = np.sort(row)
    top_6_elements = sorted_row[-6:]
    # result_row = np.append(top_6_elements, temp_removed_element)
    label_set[i] = top_6_elements

print("label_set", label_set.shape)
print("label_set[0]", label_set[0])

cl_values = np.random.randint(1, 6, size=50000)

# Ensure that the average is close to 4 by adjusting individual elements
current_avg = np.mean(cl_values)

while abs(4 - current_avg) > 0.01:
    idx = np.random.randint(0, 50000)
    diff = 4 - current_avg
    if diff > 0 and cl_values[idx] < 6:
        cl_values[idx] += 1
    elif diff < 0 and cl_values[idx] > 1:
        cl_values[idx] -= 1
    current_avg = np.mean(cl_values)

# Step 4
cl_labels = np.empty((50000,), dtype=object)

for i, count in enumerate(cl_values):
    picked_numbers = np.random.choice(label_set[i], size=count, replace=False)
    cl_labels[i] = picked_numbers



for i in range(50000):
    cl_labels[i] = np.append(cl_labels[i], ground_truth_val[i])

print("cl_labels",cl_labels.shape)
print("cl_labels[0]->",cl_labels[0])
# Step 5
#pseudo_ground_truth = np.empty((50000, 1))

# Iterate over the elements in 'picked_numbers_array' and select one number from each list
#for i, picked_numbers in enumerate(cl_labels):
#    selected_number = np.random.choice(picked_numbers, size=1)
#    pseudo_ground_truth[i] = selected_number

#print("pseudo_ground_truth",pseudo_ground_truth.shape)
# Step 6
#pseudo_ground_truth_label = np.empty((50000, 1), dtype=int)

# Iterate
#for i, value in enumerate(pseudo_ground_truth):
#    row = target_set[i]
#    index = np.where(row == value)[0]
#    pseudo_ground_truth_label[i] = index

#print("pseudo_ground_truth_label", pseudo_ground_truth_label.shape)
#np.save('pseudo_ground_truth_train.npy', pseudo_ground_truth_label)
label_indices = np.zeros_like(target_set, dtype=int)

for i, picked_numbers in enumerate(cl_labels):
    indices = np.where(np.isin(target_set[i], picked_numbers))
    label_indices[i, indices] = 1

print("label_indices[0]->",label_indices[0])
np.save('labels_train.npy', label_indices)




#test

model_trained.eval()
test_features = []
test_predictions = []
test_targets = []
with torch.no_grad():
    for inputs, target in test_loader:
        inputs, target = inputs.to(device), target.to(device)
        outputs = model_trained(inputs)
        test_predictions.append(outputs.cpu())
        # print(inputs.shape)
        test_features.append(inputs.cpu().numpy())
        test_targets.append(target.cpu().numpy())

test_features_np = np.concatenate(test_features, axis=0)
test_label_set = np.concatenate(test_predictions, axis=0)
test_ground_truth = np.concatenate(test_targets, axis=0)

np.save('./test_features.npy', test_features_np)
np.save('./test_label_set.npy', test_label_set)
np.save('./test_ground_truth.npy', test_ground_truth)


# Step 1
target_set_ = np.load('test_label_set.npy')
true_label_ = np.load('test_ground_truth.npy')

print("target_set_test[0]",target_set_[0])
print("true_label_test[0]",true_label_[0])

print("Target_set_",target_set_.shape)
print("True_label_",true_label_.shape)
# Step 2
label_set_ = np.empty((10000, 6))
ground_truth_val_ = np.empty((10000, 1))
# Iterate
for i in range(10000):
    row = target_set_[i]  # Corrected variable name
    index_to_remove = true_label_[i]  # Corrected variable name
    ground_truth_val_[i] = row[index_to_remove]
    row = np.delete(row, index_to_remove)
    sorted_row = np.sort(row)
    top_6_elements = sorted_row[-6:]
    # result_row = np.append(top_6_elements, temp_removed_element)
    label_set_[i] = top_6_elements

print("label_set_", label_set_.shape)
print("label_set_[0]", label_set_[0])

cl_values_ = np.random.randint(1, 6, size=10000)

# Ensure that the average is close to 4 by adjusting individual elements
current_avg_ = np.mean(cl_values_)

while abs(4 - current_avg_) > 0.01:
    idx = np.random.randint(0, 10000)
    diff = 4 - current_avg_
    if diff > 0 and cl_values_[idx] < 6:
        cl_values_[idx] += 1
    elif diff < 0 and cl_values_[idx] > 1:
        cl_values_[idx] -= 1
    current_avg_ = np.mean(cl_values_)

# Step 4
cl_labels_ = np.empty((10000,), dtype=object)

for i, count in enumerate(cl_values_):
    picked_numbers = np.random.choice(label_set_[i], size=count, replace=False)
    cl_labels_[i] = picked_numbers

for i in range(10000):
    cl_labels_[i] = np.append(cl_labels_[i], ground_truth_val_[i])  # Corrected variable name

print("cl_labels_",cl_labels_.shape)
print("cl_labels_[0]->",cl_labels_[0])

# Step 5
pseudo_ground_truth_ = np.empty((10000, 1))

# Iterate over the elements in 'picked_numbers_array' and select one number from each list
#for i, picked_numbers in enumerate(cl_labels_):
#    selected_number = np.random.choice(picked_numbers, size=1)
#    pseudo_ground_truth_[i] = selected_number

#print("Pseudo ground_truth_",pseudo_ground_truth_.shape)

# Step 6
#pseudo_ground_truth_label_ = np.empty((10000, 1), dtype=int)

# Iterate
#for i, value in enumerate(pseudo_ground_truth_):
#    row = target_set_[i]  # Corrected variable name
#    index = np.where(row == value)[0]
#    pseudo_ground_truth_label_[i] = index

#np.save('pseudo_ground_truth_test.npy', pseudo_ground_truth_label_)
label_indices_ = np.zeros_like(target_set_, dtype=int)

for i, picked_numbers in enumerate(cl_labels_):
    indices_ = np.where(np.isin(target_set_[i], picked_numbers))
    label_indices_[i, indices_] = 1
print("label_indices_[0]->",label_indices_[0])
np.save('labels_test.npy', label_indices_)
