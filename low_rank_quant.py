"""
Load a trained autoencoder in fp32 
Check if the autoencoder can be compressed using concatenated low rank factorization methods
"""


import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from net.models import LeNet, AE
from net.quantization import apply_weight_sharing
import util

os.makedirs('saves', exist_ok=True)

# def calc_model_size(r,):

def plot_mat(low_rank_thresh, model, id):
    mat_enc = util.to_cpu_data(model, 2*id)
    mat_dec = util.to_cpu_data(model,10-2*id)
    u1, s1, vh1 =np.linalg.svd(mat_enc)
    u2, s2, vh2 =np.linalg.svd(mat_dec)
    mat = np.concatenate((mat_enc, mat_dec.T))
    u, s, vh = np.linalg.svd(mat)
    r, _ = rank_selector(low_rank_thresh[id], mat)
    mat_r = np.matmul(np.matmul(u[:,0:r], np.diag(s[0:r])), vh[0:r,:])
    mat_r_enc = mat_r[0:mat_enc.shape[0],:]
    mat_r_dec = mat_r[mat_enc.shape[0]:,:].T
    plot_svd(mat_enc,"Singular Values of Encoder " + str(id) + " Weight Matrix")
    plt.savefig("results/Encoder"+str(id)+"_SVD.png")
    plot_svd(mat_dec,"Singular Values of ccDecoder " + str(id) + " Weight Matrix")
    plt.savefig("results/Decoder"+str(id)+"_SVD.png")
    plot_svd(mat, "Singular Values of Concatenated Encoder and Decoder " + str(id))
    plt.savefig("results/Concat"+str(id)+"_SVD.png")
    # print(np.linalg.norm(mat_enc - mat_r_enc, 'fro'))
    

def plot_svd(mat, title):
    u,s,vh = np.linalg.svd(mat)
    plt.figure()
    plt.plot(s)
    plt.title(title)
    plt.show()
def test_vae(model):
    # print("invoking test_vae")
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
        # print(f'Test set: Average loss: {total_loss:.4f}')
    return total_loss / len(test_loader.dataset)


def rank_selector(threshold, mat):
    rank_err = []
    u, s, vh = np.linalg.svd(mat)
    for r in range(2,mat.shape[0]):
        mat_r = np.matmul(np.matmul(u[:,0:r], np.diag(s[0:r])), vh[0:r,:])
        err = np.linalg.norm(mat - mat_r,'fro')/np.linalg.norm(mat,'fro')
        rank_err.append(err)
        if err < threshold:
            break
    # print(r)
    return r, rank_err


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--symm_pruning', type=bool, default=False,
                    help="whether to use the same pruning thresholds for encoder and decoder layers")
parser.add_argument('--architecture', type=str, default='lenet',
                    help="which architecture to prune. VAE and lenet supported")
parser.add_argument('--save_name', type=str, default='',
                    help="name of architecture to be saved")
parser.add_argument('--eps_arr', type=float, nargs=3, metavar=(1e-1, 1e-1, 1e-1),
                    help="array of thresholds for low rank approximation")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)



model = torch.load('saves/initial_model_vaelrq.ptmodel')
# model = torch.load('saves/initial_model_vaevae_wd_e-5.ptmodel')
print(model)  
model_comp = torch.load('saves/initial_model_vaelrq.ptmodel') #AE(mask=False, input_shape=784).to(device)
model_comp_ind = torch.load('saves/initial_model_vaelrq.ptmodel')
# util.print_model_weight_values(model)
low_rank_thresh = args.eps_arr #[1e-1, 1e-1, 1e-1]


w = list(model.parameters())
i = 0 
def calc_lr_mat(low_rank_thresh, model, id):
    mat_enc = util.to_cpu_data(model, 2*id)
    mat_dec = util.to_cpu_data(model,10-2*id)
    u1, s1, vh1 =np.linalg.svd(mat_enc)
    u2, s2, vh2 =np.linalg.svd(mat_dec)
    mat = np.concatenate((mat_enc, mat_dec.T))
    u, s, vh = np.linalg.svd(mat)
    r, _ = rank_selector(low_rank_thresh[id], mat)
    mat_r = np.matmul(np.matmul(u[:,0:r], np.diag(s[0:r])), vh[0:r,:])
    mat_r_enc = mat_r[0:mat_enc.shape[0],:]
    mat_r_dec = mat_r[mat_enc.shape[0]:,:].T
    # print(np.linalg.norm(mat_enc - mat_r_enc, 'fro'))
    return r, mat_r_enc, mat_r_dec
def calc_lr_ind_mat(low_rank_thresh, model, id):
    mat_enc = util.to_cpu_data(model, 2*id)
    mat_dec = util.to_cpu_data(model,10-2*id)
    u1, s1, vh1 =np.linalg.svd(mat_enc)
    u2, s2, vh2 =np.linalg.svd(mat_dec)
    r_enc, _ = rank_selector(low_rank_thresh[id]/np.sqrt(4), mat_enc)
    r_dec, _ = rank_selector(low_rank_thresh[id]/np.sqrt(4), mat_dec)
    mat_r_enc = np.matmul(np.matmul(u1[:,0:r_enc], np.diag(s1[0:r_enc])), vh1[0:r_enc,:])
    mat_r_dec = np.matmul(np.matmul(u2[:,0:r_dec], np.diag(s2[0:r_dec])), vh2[0:r_dec,:])
    # print(np.linalg.norm(mat_enc - mat_r_enc, 'fro'))
    # print(np.linalg.norm(mat_dec - mat_r_dec, 'fro'))
    return r_enc, r_dec, mat_r_enc, mat_r_dec
# Joint
with torch.no_grad():
    r1, mat_r_enc, mat_r_dec = calc_lr_mat(low_rank_thresh, model, 0)
    model_comp.encoder_l1.weight.data = torch.tensor(mat_r_enc)
    model_comp.decoder_l1.weight.data = torch.tensor(mat_r_dec)
    # plot_mat(low_rank_thresh, model, 0)
    
    r2, mat_r_enc, mat_r_dec = calc_lr_mat(low_rank_thresh, model, 1)
    model_comp.encoder_l2.weight.data = torch.tensor(mat_r_enc)
    model_comp.decoder_l2.weight.data = torch.tensor(mat_r_dec)
    # plot_mat(low_rank_thresh, model, 1)

    r3, mat_r_enc, mat_r_dec = calc_lr_mat(low_rank_thresh, model, 2)
    model_comp.encoder_l3.weight.data = torch.tensor(mat_r_enc)
    model_comp.decoder_l3.weight.data = torch.tensor(mat_r_dec)
    # plot_mat(low_rank_thresh, model, 2)
# Individual
with torch.no_grad():
    r_enc_1, r_dec_1, mat_r_enc, mat_r_dec = calc_lr_ind_mat(low_rank_thresh, model, 0)
    model_comp_ind.encoder_l1.weight.data = torch.tensor(mat_r_enc)
    model_comp_ind.decoder_l1.weight.data = torch.tensor(mat_r_dec)
    
    r_enc_2, r_dec_2, mat_r_enc, mat_r_dec = calc_lr_ind_mat(low_rank_thresh, model, 1)
    model_comp_ind.encoder_l2.weight.data = torch.tensor(mat_r_enc)
    model_comp_ind.decoder_l2.weight.data = torch.tensor(mat_r_dec)
    
    r_enc_3, r_dec_3, mat_r_enc, mat_r_dec = calc_lr_ind_mat(low_rank_thresh, model, 2)
    model_comp_ind.encoder_l3.weight.data = torch.tensor(mat_r_enc)
    model_comp_ind.decoder_l3.weight.data = torch.tensor(mat_r_dec)

# plt.figure()
# plt.plot(s1)
# plt.figure()
# plt.plot(s2)
# plt.figure()
# plt.plot(s)
# plt.show()

util.print_model_parameters(model)
util.print_model_parameters(model_comp)

print("Rank selected for joint: r1, r2, r3", (r1,r2, r3))
print("Rank selected for Ind: (r_enc_1, r_dec_1) , (r_enc_2, r_dec_2), (r_enc_3, r_dec_3)",
    (r_enc_1, r_dec_1, r_enc_2, r_dec_2, r_enc_3, r_dec_3))

print("eps vals: ", args.eps_arr)
# print("original model error:")
acc_org = test_vae(model.to(device))*1e4
# print("joint compressed model error:")
acc_joint_comp = test_vae(model_comp.to(device))*1e4
# print("Individually compressed model error:")
acc_ind_comp = test_vae(model_comp_ind.to(device))*1e4
model_dummy = AE(mask=False, input_shape=784).to(device)
acc_random = test_vae(model_dummy)*1e4 

params = {}

params['rj1'] = r1; params['rj2'] = r2; params['rj3'] = r3;
params['ri_enc_1'] = r_enc_1; params['ri_enc_2'] = r_enc_2; params['ri_enc_3'] = r_enc_3; 
params['ri_dec_1'] = r_dec_1; params['ri_dec_2'] = r_dec_2; params['ri_dec_3'] = r_dec_3

util.low_rank_comp(params)
print(f'acc_org: {acc_org:.3f},acc_ind_comp: ,{acc_ind_comp:.3f}, acc_joint_comp: , {acc_joint_comp:.3f}, acc_random: , {acc_random: .3f}')
   

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
initial_optimizer_state_dict = optimizer.state_dict()
# print(s.shape)
def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if args.architecture != 'lenet':
                loss = F.mse_loss(output, data.view(-1,784))
            else:
                loss = F.nll_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


