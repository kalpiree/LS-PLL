'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from test__ import initialize_ground_truth_labels, update_smoothed_labels, calculate_weighted_term_mean, \
    update_ground_truth_labels
import os
import argparse
import resnet as RN
from utils import progress_bar, LabelSmoothingCrossEntropy, save_model

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ce', action='store_true', help='Cross entropy use')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 120

# Data
print('==> Preparing data..')
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

# Assuming you have your model, optimizer, criterion, and other settings defined

# Example data (replace these with your actual tensors)
input_numpy = np.load('train_features.npy')
input_tensor = torch.from_numpy(input_numpy)

label_set_numpy = np.load('labels_train.npy')
label_set_tensor = torch.from_numpy(label_set_numpy).to(torch.float64)

softmax_accumulator = torch.zeros(label_set_tensor.shape, dtype=torch.float64)

targets_tensor, label_set_tensor = initialize_ground_truth_labels(label_set_tensor)
output_tensor = torch.zeros(label_set_tensor.shape, dtype=torch.float64)

# targets_tensor = torch.randint(0, 2, (50000, 1))

# Combine input_tensor, label_set_tensor, and targets_tensor into a list of tuples
data_tuples = list(zip(input_tensor, label_set_tensor, softmax_accumulator, targets_tensor, output_tensor))

# Model
print('==> Building model..')
net = RN.ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.ce == True:
    criterion = nn.CrossEntropyLoss()
    save_path = './checkpoint/CrossEntropy.bin'
    print("Use CrossEntropy")
else:
    criterion = calculate_weighted_term_mean
    save_path = './checkpoint/LabelSmoothing.bin'
    print("Use Label Smooting")

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

# Example model, optimizer, criterion (replace these with your actual model, optimizer, criterion)

# Number of epochs
# num_epochs = 10
batch_size = 256
smoothing_rate = 0.1


def train(epoch, data_tuples):
    # Training loop
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    # Shuffle the data_tuples at the beginning of each epoch
    random_order = torch.randperm(len(data_tuples))
    shuffled_data = [data_tuples[i] for i in random_order]
    batch_index = 0
    # Iterate over batches
    for batch_start in range(0, len(data_tuples), batch_size):
        batch_index += 1
        print("Batch_index->",batch_index)
        batch_end = min(batch_start + batch_size, len(data_tuples))
        batch = shuffled_data[batch_start:batch_end]
        indices = random_order[batch_start:batch_end]
        inputs, label_set_gpu, softmax_accu_gpu, targets_gpu, outputs = zip(*batch)
        inputs = torch.stack(inputs).to(device)
        label_set_gpu = torch.stack(label_set_gpu).to(device)
        softmax_accu_gpu = torch.stack(softmax_accu_gpu).to(device)
        #softmax_accu_gpu = torch.cat(softmax_accu_gpu, dim=0).to(device)
        targets_gpu = torch.stack(targets_gpu).to(device)
        #print("Before smoothing labels")
        label_set_gpu = update_smoothed_labels(label_set_gpu, targets_gpu, smoothing_rate)
        optimizer.zero_grad()
        # Combine inputs and updated label_set_tensor into a single tensor if needed
        # combined_inputs = torch.cat([inputs, label_set], dim=1)

        outputs = net(inputs)
        #print("Before the objective function")
        # Smoothed label will also go as input to the criterion
        loss = criterion(label_set_gpu, outputs)

        #print("Before ground truth updation")
        softmax_accu_gpu_updated = softmax_accu_gpu.clone()
        targets_gpu, softmax_accu_gpu_updated = update_ground_truth_labels(outputs, label_set_gpu, softmax_accu_gpu_updated,
                                                                   weighting_parameter=0.1)
        #print("After ground truth updation")

        loss.backward()
        optimizer.step()

        # # Move tensors back to the CPU for updating
        # label_set = torch.stack(label_set).to(device)
        # softmax_accu = torch.stack(label_set).to(device)
        # targets_tensor[indices] = targets.cpu()

        train_loss += loss.item()
        
        label_set_cpu = label_set_gpu.cpu()
        softmax_accu_cpu = softmax_accu_gpu_updated.cpu()
        softmax_accu_cpu = softmax_accu_gpu.cpu()
        targets_cpu = targets_gpu.cpu()

        label_set_tensor[indices] = label_set_cpu
        softmax_accumulator[indices] = softmax_accu_cpu
        targets_tensor[indices] = targets_cpu

        if batch_index % 5 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, 10, batch_index, len(data_tuples) // batch_size, (train_loss / batch_index)))
    data_tuples = list(zip(input_tensor.cpu(),  ## changed the device
                           label_set_tensor,  # Convert to numpy arrays if needed
                           softmax_accumulator,
                           targets_tensor, output_tensor.cpu()
                           ))

    # data_tuples = list(zip(input_tensor.to(device),          ## changed the device just check if I need to send it to any device or not and see if needed or not
    #                        label_set_tensor.to(device),  # Convert to numpy arrays if needed
    #                        softmax_accumulator.to(device),
    #                        targets_tensor.cpu(), output_tensor.to(device)
    #                        ))

    scheduler.step()
    return data_tuples


input_numpy_ = np.load('test_features.npy')
input_tensor_ = torch.from_numpy(input_numpy_)

gt_numpy_ = np.load('test_ground_truth.npy')
gt_tensor_ = torch.from_numpy(gt_numpy_)
print(gt_tensor_.dtype)

# output_tensor_ = torch.zeros((label_set_tensor.shape[0], 10), dtype=torch.float64)
# gt_obtained_tensor_ = torch.zeros_like(gt_tensor_)

data_tuples_ = list(zip(input_tensor_, gt_tensor_))

batch_size_ =100
def test(epoch):
    # Training loop
    print('\nEpoch: %d' % epoch)
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    net.eval()
    batch_index_ = 0

    # # Shuffle the data_tuples at the beginning of each epoch
    # random_order = torch.randperm(len(data_tuples_))
    # shuffled_data_ = [data_tuples_[i] for i in random_order]
    with torch.no_grad():
        # Iterate over batches
        for batch_start in range(0, len(data_tuples_), batch_size_):
            batch_index_ += 1
            batch_end = min(batch_start + batch_size_, len(data_tuples_))
            batch = data_tuples_[batch_start:batch_end]
            # indices = random_order[batch_start:batch_end]
            inputs_, gt = zip(*batch)
            inputs_ = torch.stack(inputs_).to(device)
            gt = torch.stack(gt).to(device)
            output_ = net(inputs_)
            _, gt_obtained = output_.max(1)
            print(gt_obtained.dtype)

            total += gt.size(0)
            correct += gt_obtained.eq(gt).sum().item()

            progress_bar(batch_index_, len(data_tuples_),
                         'Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

            # Combine inputs and updated label_set_tensor into a single tensor if needed
            # combined_inputs = torch.cat([inputs, label_set], dim=1)
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_model(net, save_path)
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 500):
    data_tuples = train(epoch, data_tuples)
    test(epoch)
