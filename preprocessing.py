# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import time
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import lenet as LN
import resnet as RN
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
device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


print('==> Building model..')
net = LN.LeNet5Model()
#model =net()
#net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0001, nesterov= True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # You can adjust the angle as needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train,)   # set download = False if folder and dataset exists
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)


def evaluate_model(net, test_loader, criterion):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, ground_truth_label) in enumerate(test_loader):
            inputs, ground_truth_label = inputs.to(device), ground_truth_label.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, ground_truth_label)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += ground_truth_label.size(0)
            correct += predicted.eq(ground_truth_label).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss / (batch_idx + 1):.3f}, Test Accuracy: {accuracy:.2f}%")

    return accuracy

def train_model_with_test(net, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=5):
    best_accuracy = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, ground_truth_label) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, ground_truth_label = inputs.to(device), ground_truth_label.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, ground_truth_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += ground_truth_label.size(0)
            correct += predicted.eq(ground_truth_label).sum().item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.3f}, Train Accuracy: {100. * correct / total:.2f}%")
        scheduler.step()

        # Evaluate on the test set
        test_accuracy = evaluate_model(net, test_loader, criterion)

        # Save the model if it has the best test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = net.state_dict()
            print("Saving/updating best model...")
            torch.save(best_model_state, 'best_model.pth')
        print("Best accuracy is:",best_accuracy)

    # Load the best model's state dictionary
    net.load_state_dict(best_model_state)

    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

    return net

# Create DataLoaders for training and testing
train_loader = DataLoader(trainset, batch_size=128, shuffle=True,num_workers=10)
test_loader = DataLoader(testset, batch_size=128, shuffle=False,num_workers=10)
model_trained = train_model_with_test(net, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=75)

train_loader = DataLoader(trainset, batch_size=128, shuffle=False)
train_features = []
train_predictions = []
train_targets = []

with torch.no_grad():
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)
        outputs = model_trained(inputs)
        train_predictions.append(outputs.cpu())
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

# Step 2
label_set = np.empty((60000, 6))                # set values for the dataset size and the prediction set size

ground_truth_val = np.empty((60000, 1))
# Iterate
for i in range(60000):
    row = target_set[i]
    index_to_remove = true_label[i]
    ground_truth_val[i] = row[index_to_remove]
    row = np.delete(row, index_to_remove)
    sorted_row = np.sort(row)
    top_6_elements = sorted_row[-6:]
    label_set[i] = top_6_elements


cl_values = np.random.randint(0, 6, size=60000)

current_avg = np.mean(cl_values)

print("Current avg",current_avg)

while abs(2 - current_avg) > 0.01:               # set avg according to #avg_cl value
    idx = np.random.randint(0, 60000)
    diff = 2 - current_avg
    if diff > 0 and cl_values[idx] < 6:
        cl_values[idx] += 1
    elif diff < 0 and cl_values[idx] > 0:
        cl_values[idx] -= 1
    current_avg = np.mean(cl_values)

print("Avg after loop", current_avg)

# Step 4
cl_labels = np.empty((60000,), dtype=object)

for i, count in enumerate(cl_values):
    picked_numbers = np.random.choice(label_set[i], size=count, replace=False)
    cl_labels[i] = picked_numbers

for i in range(60000):
    cl_labels[i] = np.append(cl_labels[i], ground_truth_val[i])


# Step 5
label_indices = np.zeros_like(target_set, dtype=int)

for i, picked_numbers in enumerate(cl_labels):
    indices = np.where(np.isin(target_set[i], picked_numbers))
    label_indices[i, indices] = 1

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

# Step 2                                                            # set values for the dataset size and the prediction set size
label_set_ = np.empty((10000, 6))
ground_truth_val_ = np.empty((10000, 1))

for i in range(10000):
    row = target_set_[i]  
    index_to_remove = true_label_[i]  
    ground_truth_val_[i] = row[index_to_remove]
    row = np.delete(row, index_to_remove)
    sorted_row = np.sort(row)
    top_6_elements = sorted_row[-6:]
    label_set_[i] = top_6_elements


cl_values_ = np.random.randint(0, 6, size=10000)

current_avg_ = np.mean(cl_values_)

while abs(2 - current_avg_) > 0.01:                                   # set avg according to #avg_cl value
    idx = np.random.randint(0, 10000)
    diff = 2 - current_avg_
    if diff > 0 and cl_values_[idx] < 6:
        cl_values_[idx] += 1
    elif diff < 0 and cl_values_[idx] > 0:
        cl_values_[idx] -= 1
    current_avg_ = np.mean(cl_values_)

# Step 4
cl_labels_ = np.empty((10000,), dtype=object)

for i, count in enumerate(cl_values_):
    picked_numbers = np.random.choice(label_set_[i], size=count, replace=False)
    cl_labels_[i] = picked_numbers

for i in range(10000):
    cl_labels_[i] = np.append(cl_labels_[i], ground_truth_val_[i])  # Corrected variable name


# Step 5


label_indices_ = np.zeros_like(target_set_, dtype=int)

for i, picked_numbers in enumerate(cl_labels_):
    indices_ = np.where(np.isin(target_set_[i], picked_numbers))
    label_indices_[i, indices_] = 1

np.save('labels_test.npy', label_indices_)
