import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)

def low_rank_comp(params):
    rj = [params['rj1'], params['rj2'], params['rj3']]
    ri_enc = [params['ri_enc_1'],  params['ri_enc_2'], params['ri_enc_3']]
    ri_dec = [params['ri_dec_1'], params['ri_dec_2'], params['ri_dec_3']]

    lsize = [784, 256, 128, 64]
    true_model_params = 0
    joint_model_params = 0
    ind_model_params = 0
    for i in range(len(lsize) - 1):
        true_model_params = true_model_params + lsize[i]*lsize[i+1]
        joint_model_params = joint_model_params + rj[i]*(lsize[i] + 2*lsize[i+1] + 1)
        ind_model_params = ind_model_params + (ri_enc[i] + ri_dec[i])*(lsize[i] + lsize[i+1] + 1)
    true_model_params = true_model_params*2
    print(f'true_model_params: {true_model_params:.3f},joint compression: ,{true_model_params/joint_model_params:.3f}x, ind compression: , {true_model_params/ind_model_params:.3f}x')



def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)

def print_model_weight_values(model):
    w = list(model.parameters())
    for i in w:
        print(i.detach().cpu().numpy().shape)

def to_cpu_data(model, id):
    w = list(model.parameters())
    return w[id].detach().cpu().numpy()


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def test_vae(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    print("invoking test_vae")
    model.eval()
    test_loss = 0
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = F.mse_loss(output, data.view(-1,784)).item() # sum up batch loss
            total_loss = total_loss + test_loss
            # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).sum().item()
        # total_loss /= len(test_loader.dataset)
        # accuracy = 100. * correct / len(test_loader.dataset)
        print(len(test_loader.dataset))
        print(f'Test set: Average loss: {total_loss:.4f}')
    return total_loss / len(test_loader.dataset)
