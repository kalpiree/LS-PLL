import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from algorithm_fn import initialize_ground_truth_labels, update_smoothed_labels, calculate_weighted_term_mean, \
    update_ground_truth_labels
import os
import argparse
import lenet as LN
from utils import progress_bar, LabelSmoothingCrossEntropy, save_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

#change the description according to the dataset
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ce', action='store_true', help='Cross entropy use')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

#file obtained from the preprocessing step
input_numpy = np.load('train_features.npy')
input_tensor = torch.from_numpy(input_numpy)

label_set_numpy = np.load('labels_train.npy')
label_set_tensor = torch.from_numpy(label_set_numpy).to(torch.float64)

softmax_accumulator = torch.zeros(label_set_tensor.shape, dtype=torch.float64)

targets_tensor, label_set_tensor = initialize_ground_truth_labels(label_set_tensor)
output_tensor = torch.zeros(label_set_tensor.shape, dtype=torch.float64)


# Combine input_tensor, label_set_tensor, and targets_tensor into a list of tuples
data_tuples = list(zip(input_tensor, label_set_tensor, softmax_accumulator, targets_tensor, output_tensor))

# Model
print('==> Building model..')
net = LN.LeNet5Model()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.ce == True:
    criterion = nn.CrossEntropyLoss()
    save_path = './checkpoint/CrossEntropy_7_e_2_ls_cl_3.bin'                       #set the path of the saved model                      
    print("Use CrossEntropy")
else:
    criterion = calculate_weighted_term_mean
    save_path = './checkpoint/LabelSmoothing_0_e_1_ls_cl_3.bin'                     #set the path of the saved model
    print("Use Label Smooting")


optimizer_sgd = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001, nesterov=True)
scheduler_sgd = ReduceLROnPlateau(optimizer_sgd, mode='min', factor=0.5, patience=3, verbose=True)


optimizer = optimizer_sgd  
scheduler = scheduler_sgd  

batch_size = 128
smoothing_rate = 0.0                                                                    # set the smoothing rate
print("The smoothing rate used is:",smoothing_rate)

train_losses_file = 'train_losses_0_e_1_ls_cl_3.npy'                                    #  create the numpy file for train losses
train_losses = []


def train(epoch, data_tuples,train_losses):
    # Training loop
    print("Epoch no:", epoch+1)
    net.train()
    train_loss = 0

    # Shuffle the data_tuples at the beginning of each epoch
    random_order = torch.randperm(len(data_tuples))
    shuffled_data = [data_tuples[i] for i in random_order]
    batch_index = 0
    # Iterate over batches
    for batch_start in range(0, len(data_tuples), batch_size):
        batch_index += 1
        batch_end = min(batch_start + batch_size, len(data_tuples))
        batch = shuffled_data[batch_start:batch_end]
        indices = random_order[batch_start:batch_end]
        inputs, label_set_gpu, softmax_accu_gpu, targets_gpu, outputs = zip(*batch)
        inputs = torch.stack(inputs).to(device)
        label_set_gpu = torch.stack(label_set_gpu).to(device)
        softmax_accu_gpu = torch.stack(softmax_accu_gpu).to(device)
        targets_gpu = torch.stack(targets_gpu).to(device)
        label_set_gpu = update_smoothed_labels(label_set_gpu, targets_gpu, smoothing_rate)
        optimizer.zero_grad()


        outputs = net(inputs)
        loss = criterion(label_set_gpu, outputs)
        softmax_accu_gpu_updated = softmax_accu_gpu.clone()
        targets_gpu, softmax_accu_gpu_updated = update_ground_truth_labels(outputs, label_set_gpu, softmax_accu_gpu_updated,
                                                                   weighting_parameter=0.9)

        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        
        label_set_cpu = label_set_gpu.cpu()
        softmax_accu_cpu = softmax_accu_gpu_updated.cpu()
        softmax_accu_cpu = softmax_accu_gpu.cpu()
        targets_cpu = targets_gpu.cpu()

        label_set_tensor[indices] = label_set_cpu
        softmax_accumulator[indices] = softmax_accu_cpu
        targets_tensor[indices] = targets_cpu

        if batch_index % 50 == 0:
            print("Batch index:",batch_index)
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, 100, batch_index, len(data_tuples) // batch_size, (train_loss / batch_index)))

            
    data_tuples = list(zip(input_tensor.cpu(),  
                           label_set_tensor,  
                           softmax_accumulator,
                           targets_tensor, output_tensor.cpu()
                           ))

    train_losses.append(train_loss/batch_index)
    print("train_losses",train_losses)
    scheduler.step(train_loss/batch_index)
    print(f"Learning rate:{optimizer.param_groups[0]['lr']}")
    return data_tuples



test_accuracy_file = 'test_accuracy_0_e_1_ls_cl_3.npy'                                          #create the test accuracy numpy file
test_accuracy =[]

input_numpy_ = np.load('test_features.npy')
input_tensor_ = torch.from_numpy(input_numpy_)

gt_numpy_ = np.load('test_ground_truth.npy')
gt_tensor_ = torch.from_numpy(gt_numpy_)
data_tuples_ = list(zip(input_tensor_, gt_tensor_))

batch_size_ =100
def test(epoch,test_accuracy):
    print('Epoch:',epoch+1)
    global best_acc
    correct = 0
    total = 0
    net.eval()
    batch_index_ = 0
    
    with torch.no_grad():
        for batch_start in range(0, len(data_tuples_), batch_size_):
            batch_index_ += 1
            batch_end = min(batch_start + batch_size_, len(data_tuples_))
            batch = data_tuples_[batch_start:batch_end]
            inputs_, gt = zip(*batch)
            inputs_ = torch.stack(inputs_).to(device)
            gt = torch.stack(gt).to(device)
            output_ = net(inputs_)
            _, gt_obtained = output_.max(1)

            total += gt.size(0)
            correct += gt_obtained.eq(gt).sum().item()

            progress_bar(batch_index_, len(data_tuples_),
                         'Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

    acc = 100. * correct / total
    test_accuracy.append(acc)
    print("Accuracy after Epoch ->",epoch+1,":",acc)
    print("Test accuracy value ->",test_accuracy)
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_model(net, save_path)
    
    print("Best accuracy is:",best_acc)


for epoch in range(start_epoch, start_epoch + 100):
    data_tuples = train(epoch, data_tuples,train_losses)
    test(epoch,test_accuracy)


# Save the final list of training losses after all epochs
np.save(train_losses_file, np.array(train_losses))
print("Final train_losses",train_losses)
np.save(test_accuracy_file, np.array(test_accuracy))
print("Final train_accuracies", test_accuracy)
