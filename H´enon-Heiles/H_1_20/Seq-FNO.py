import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import os
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

print(torch.__version__)

if not os.path.exists('results/FNO_3/model/'):
    os.makedirs('results/FNO_3/model/')
if not os.path.exists('results/FNO_3/plot/'):
    os.makedirs('results/FNO_3/plot/')
if not os.path.exists('results/FNO_3/pred/'):
    os.makedirs('results/FNO_3/pred/')

path_model= 'results/FNO_3/model/'
path_plot= 'results/FNO_3/plot/'
path_pred= 'results/FNO_3/pred/'
################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):  # x:[20,64,1024]
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        # print('1111111',x_ft.shape) [20,64,513]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2


        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  configurations
################################################################
ntrain =1000
ntest = 100

s = 4

batch_size = 20
learning_rate = 0.001

epochs =1000
step_size = 50
gamma = 0.5

modes = 2
width = 128 #64

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('50_step_pred/data_ti.mat')
x_data = dataloader.read_field('data')

y_data_list = []
y_next_train=[]
y_next_test=[]
# 循环读取每个 .mat 文件
for i in range(1, 3): #[1,2]
    file_name = f'50_step_pred/data_next_{i}.mat'
    dataloader = MatReader(file_name)
    y_data = dataloader.read_field('data')
    y_data_list.append(y_data)

x_train = x_data[:ntrain, :]
x_test = x_data[-ntest:, :]
for i in range(2):
    temp=y_data_list[i]
    y_next_train.append(temp[:ntrain,:])
    y_next_test.append(temp[-ntest:,:])

x_train = x_train.reshape(ntrain, s, 1)
x_test = x_test.reshape(ntest, s, 1)

y_train=torch.hstack(y_next_train)#(1000,8)
y_test=torch.hstack((y_next_test))#(1000,8)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width).cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

# for ep in range(epochs):
#     model.train()
#     t1 = default_timer()
#     train_mse = 0
#     train_l2 = 0
#     for x, y in train_loader:
#         x, y= x.cuda(), y.cuda()
#         optimizer.zero_grad()
#
#         outputs = {}# 创建一个字典来保存每次的输出
#         out = x
#         for i in range(1,3):
#             out_next = model(out)
#             outputs[f'out{i}'] = out_next
#             out = out_next.reshape(batch_size, s, 1)  # reshaping 每次输出
#
#         l2=torch.tensor(0.0).cuda()
#         for i in range(1,3):  # 计算所有的损失
#             out_i = outputs[f'out{i}']
#             y_i=y[:,(4*i-4):4*i]
#
#             l2 += myloss(out_i.view(batch_size, -1), y_i.view(batch_size, -1))
#         l2.backward() # use the l2 relative loss
#
#         optimizer.step()
#         train_l2 += l2.item()
#
#     scheduler.step()
#     model.eval()
#     test_l2 = 0.0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.cuda(), y.cuda()
#
#             outputs = {}  # 创建一个字典来保存每次的输出
#             out = x
#             for i in range(1, 3):
#                 out_next = model(out)
#                 outputs[f'out{i}'] = out_next
#                 out = out_next.reshape(batch_size, s, 1)  # reshaping 每次输出
#             for i in range(1, 3):  # 计算所有的损失
#                 out_i = outputs[f'out{i}']
#                 y_i = y[:, (4 * i -4):4 * i]
#                 test_l2 += myloss(out_i.view(batch_size, -1), y_i.view(batch_size, -1)).item()
#
#     train_mse /= len(train_loader)
#     train_l2 /= ntrain
#     test_l2 /= ntest
#
#     t2 = default_timer()
#     print("Epoch: %d, time: %.3f, Train l2:%.4f  Test l2: %.4f"
#                   % ( ep, t2-t1, train_l2, test_l2) )
# torch.save(model, path_model+'FNO_3')

model = torch.load(path_model+'FNO_3')
pred = torch.zeros(ntest,4)
index = 0
test_e = torch.zeros(y_test.shape[0])
test_mse_list = torch.zeros(y_test.shape[0])


dataloadera = MatReader('50_step_pred/IC.mat')
dataloaderu = MatReader('50_step_pred/muti_time/u_1.0.mat')
x_data = dataloadera.read_field('ic')[:,]
y_data = dataloaderu.read_field('u1.00')[:,]

x_test = x_data[-ntest:, :]
y_test = y_data[-ntest:, :]

x_test = x_test.reshape(ntest, s, 1)
test_l2_list=torch.zeros(ntest)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=1, shuffle=False)
pred_error_list = np.zeros(50)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()
        out = model(x).view(-1) #64
        #print('22222',out.shape)
        pred[index] = out
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()

        test_l2_list[index]=test_l2
        index = index + 1

    print('Mean Error:', round(100 * torch.mean(test_l2_list).numpy(),3),'%')
    pred_error_list[0] = torch.mean(test_l2_list).numpy()
scipy.io.savemat(path_pred+'pred_1.000.mat',{'pred': pred.numpy()})


#pred next timeslice
for i in range(49):
    dataloaderu = MatReader(path_pred+'pred_{:.3f}.mat'.format(1.000+i*1.000))
    dataloader = MatReader('50_step_pred/muti_time/u_{:.1f}'.format(2+i*1))

    y_data = dataloaderu.read_field('pred')[:,]
    z_data = dataloader.read_field('u{:.2f}'.format(2.00+i*1.00))[:,]

    y_pred = y_data[-ntest:, :]
    z_pred = z_data[-ntest:, :]

    y_pred = y_pred.reshape(ntest, s, 1)
    pred = torch.zeros(y_test.shape)  # [100,64]
    test_e = torch.zeros(y_test.shape[0])
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_pred, z_pred), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()
            out = model(x).view(-1)
            pred[index] = out
            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            test_e[index] = test_l2
            index = index + 1
    print('Mean Error:', round(100 * torch.mean(test_e).numpy(),3),'%')
    scipy.io.savemat(path_pred+'pred_{:.3f}.mat'.format(2.000+i*1.000),{'pred': pred.numpy()})
    pred_error_list[i + 1] = torch.mean(test_e).numpy()
    print(pred_error_list[i + 1])

    plt.figure(figsize=(10, 8))
    plt.title('t=' + str(2 + i * 1))
    for j in range(y_test.shape[0]):
        if j %20==1:
            colors = ['cyan', 'r', 'fuchsia', 'darkorange', 'limegreen', 'hotpink', 'b', 'greenyellow', 'deepskyblue',
                      'yellow']
            color = colors[j // 20 % len(colors)]
            plt.plot(z_pred[j, :].numpy(), 'k', linewidth=5, alpha=0.6, label='Actual')
            plt.plot(pred[j, :].numpy(), linestyle='--', linewidth=5, label='Prediction', color=color)
        # plt.legend()
    plt.grid(True)
    plt.margins(0)
    plt.savefig(path_plot+'t={:d}.jpg'.format(2 + i * 1))
    # plt.show()
scipy.io.savemat('pred_error_ours.mat',{'pred': pred_error_list})

