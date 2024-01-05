import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import resnet as RN
import torchvision
import resnet as RN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import lenet as LN
import argparse
from dataloader_class import CIFARPseudoDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')                                         # set the dataset
    parser.add_argument('--ce', action='store_true', help='Cross Entropy use')
    args = parser.parse_args()

    test_features = np.load('./test_features.npy')
    test_label = np.load('./test_ground_truth.npy')
    test_label = test_label.reshape((10000, 1))

    model = LN.LeNet5()                                                                                                   # load the model
    if args.ce == True:
        path = './checkpoint/CrossEntropy.bin'
        npy_path = './CE.npy'
        npy_target = './CE_tar.npy'
        title = 'TSNE_CrossEntropy'
        states = torch.load(path)
    else:
        path = ('./checkpoint/LabelSmoothing_0_e_1_ls_cl_3.bin')
        npy_path = './LS.npy'
        npy_target = './LS_tar.npy'
        title = 'TSNE_LabelSmoothing_0_e_1_ls'
        states = torch.load(path)

    model.load_state_dict(states)
    model.linear = nn.Flatten()

    #transform_test = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #])
    testset = CIFARPseudoDataset(test_features, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    extract = model
    extract.eval()

    out_target = []
    out_output = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        outputs = extract(inputs)
        out_output.append(outputs.detach().numpy())
        out_target.append(targets[:, np.newaxis])

    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)

    np.save(npy_path, output_array, allow_pickle=False)
    np.save(npy_target, target_array, allow_pickle=False)

    print('Pred shape :', output_array.shape)
    print('Target shape :', target_array.shape)


    tsne = TSNE(n_components=2, init='pca', random_state=0)
    selected_output = tsne.fit_transform(output_array)

    plt.rcParams['figure.figsize'] = 10, 10
    plt.scatter(output_array[:, 0], output_array[:, 1], c=target_array[:, 0])
    plt.ylabel('Avg #CL=3', fontsize=35)
    plt.savefig('./' + title + 'cl_3'+'.png', bbox_inches='tight')







