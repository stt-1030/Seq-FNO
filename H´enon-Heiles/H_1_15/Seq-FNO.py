import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as io
from timeit import default_timer
from utilities3 import *
from Adam import Adam
torch.manual_seed(0)
np.random.seed(0)
print(torch.__version__)
M = 2 # M=1、2、3或4
result_dir = f'./results/Seq-FNO_M{M}'
path_model = result_dir + '/model/'
path_plot = result_dir + '/plot/'
path_pred = result_dir + '/pred/'
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_plot, exist_ok=True)
os.makedirs(path_pred, exist_ok=True)
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(2, self.width)
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
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    def get_grid(self, shape, device):
        batchsize = shape[0]
        size_x = shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx.to(device)

ntrain = 2000

ntest = 100
s = 4
batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 50
gamma = 0.5
modes = 2
width = 128
dataloader = MatReader('50_step_pred/data_ti.mat')
x_data = dataloader.read_field('data')
target_data = []
for j in range(1, M + 1):
    filename = f'50_step_pred/data_next_{j}_M{M}.mat'
    dataloader = MatReader(filename)
    target_data.append(dataloader.read_field('data'))
    print(f'Read supervised data: {filename}')
x_train = x_data[:ntrain, :].reshape(ntrain, s, 1)
x_test = x_data[-ntest:, :].reshape(ntest, s, 1)
target_train = [data[:ntrain, :] for data in target_data]
target_test = [data[-ntest:, :] for data in target_data]
train_dataset = torch.utils.data.TensorDataset(x_train, *target_train)
test_dataset = torch.utils.data.TensorDataset(x_test, *target_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('M:', M)
print('Training input shape:', x_train.shape)
for j in range(M):
    print(f'Supervised time index {j + 1}:', target_train[j].shape)
model = FNO1d(modes, width).cuda()
print('Number of model parameters:', count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=False)
prediction_errors = []
time_begin = time.time()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0
    for batch in train_loader:
        x = batch[0].cuda()
        targets = [value.cuda() for value in batch[1:]]
        current_batch_size = x.shape[0]
        optimizer.zero_grad()
        out = x
        l2 = 0.0
        for j in range(M):
            out = model(out)
            l2 = l2 + myloss(out.reshape(current_batch_size, -1), targets[j].reshape(current_batch_size, -1))
        l2 = l2 / M
        l2.backward()
        optimizer.step()
        train_l2 += l2.item()
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].cuda()
            targets = [value.cuda() for value in batch[1:]]
            current_batch_size = x.shape[0]
            out = x
            l2 = 0.0
            for j in range(M):
                out = model(out)
                l2 = l2 + myloss(out.reshape(current_batch_size, -1), targets[j].reshape(current_batch_size, -1))
            l2 = l2 / M
            test_l2 += l2.item()
    train_l2 /= ntrain
    test_l2 /= ntest
    t2 = default_timer()
    print('M: %d, Epoch: %d, time: %.3f, Train l2: %.6f, Test l2: %.6f' % (M, ep, t2 - t1, train_l2, test_l2))
time_end = time.time()
training_time = time_end - time_begin
print('Training time:', training_time, 'seconds')
model_file = path_model + f'Seq-FNO_M{M}.pt'
torch.save(model, model_file)
print('Model saved in:', model_file)
model.eval()
TRUE_DIR = '50_step_pred/muti_time'
dataloadera = MatReader('50_step_pred/IC.mat')
x_data = dataloadera.read_field('ic')
dataloaderu = MatReader(f'{TRUE_DIR}/u_1.0.mat')
y_data = dataloaderu.read_field('u1.00')
x_test = x_data[-ntest:, :]
y_test = y_data[-ntest:, :]
x_test = x_test.reshape(ntest, s, 1)
test_l2_list = torch.zeros(ntest)
pred = torch.zeros(ntest, s)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
index = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        out = model(x).reshape(1, -1)
        test_l2 = myloss(out, y.reshape(1, -1)).item()
        pred[index] = out.detach().cpu().reshape(-1)
        test_l2_list[index] = test_l2
        index += 1
mean_error = 100 * torch.mean(test_l2_list).item()
prediction_errors.append((1, mean_error))
print('t=1, Mean Error:', round(mean_error, 3), '%')
io.savemat(path_pred + 'pred_1.000.mat', {'pred': pred.numpy()})
pred_error_list = np.zeros(50)
pred_error_list[0] = torch.mean(test_l2_list).item()
colors = ['cyan', 'r', 'fuchsia', 'darkorange', 'limegreen', 'hotpink', 'b', 'greenyellow', 'deepskyblue', 'yellow']
T_PRED = 50
for current_time in range(2, T_PRED + 1):
    dataloader_pred = MatReader(path_pred + f'pred_{current_time:.3f}.mat'.replace(f'{current_time:.3f}', f'{current_time - 1:.3f}'))
    previous_prediction = dataloader_pred.read_field('pred')
    true_file = f'{TRUE_DIR}/u_{current_time:.1f}.mat'
    dataloader_true = MatReader(true_file)
    true_state = dataloader_true.read_field(f'u{current_time:.2f}')
    previous_prediction = previous_prediction[-ntest:, :]
    true_state = true_state[-ntest:, :]
    previous_prediction = previous_prediction.reshape(ntest, s, 1)
    pred = torch.zeros(ntest, s)
    test_errors = torch.zeros(ntest)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(previous_prediction, true_state), batch_size=1, shuffle=False)
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()
            out = model(x).reshape(1, -1)
            test_l2 = myloss(out, y.reshape(1, -1)).item()
            pred[index] = out.detach().cpu().reshape(-1)
            test_errors[index] = test_l2
            index += 1
    mean_error = 100 * torch.mean(test_errors).item()
    prediction_errors.append((current_time, mean_error))
    pred_error_list[current_time - 1] = torch.mean(test_errors).item()
    print(f't={current_time}, Mean Error: {mean_error:.3f}%')
    io.savemat(path_pred + f'pred_{current_time:.3f}.mat', {'pred': pred.numpy()})
    plt.figure(figsize=(10, 8))
    plt.title(f't={current_time}')
    for j in range(true_state.shape[0]):
        if j % 20 == 1:
            color = colors[j // 20 % len(colors)]
            plt.plot(true_state[j, :].numpy(), 'k', linewidth=5, alpha=0.6)
            plt.plot(pred[j, :].numpy(), linestyle='--', linewidth=5, color=color)
    plt.grid(True)
    plt.margins(0)
    plt.tight_layout()
    plt.savefig(path_plot + f't={current_time}.jpg', dpi=300)
    plt.close()
io.savemat(result_dir + f'/pred_error_M{M}.mat', {'pred': pred_error_list})
txt_file = result_dir + f'/results_M{M}.txt'
with open(txt_file, 'w', encoding='utf-8') as f:
    f.write(f'Training time: {training_time:.6f} seconds\n')
    for current_time, error in prediction_errors:
        f.write(f't={current_time}, Mean Error: {error:.6f}%\n')
print('Results saved in:', txt_file)
print(f'M={M} training and prediction completed.')