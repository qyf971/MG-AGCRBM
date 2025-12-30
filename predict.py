import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from utils import load_data
from model import Model
import os
from sklearn import metrics
import argparse

def mean_absolute_percentage_error(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    mape = np.abs((y_pred - y_true) / np.clip(y_true, 1e-6, None))
    return np.mean(mape[mask]) * 100


def predict_and_save_results(net, data_loader, _mean, _std, type, out_dir):
    # 每次传入batch_size=1的数据，按时刻点生成csv文件，即一组生成6个csv文件，每个csv文件包含所有传感器与预测数据和真实数据
    with torch.no_grad():
        net.eval()  # ensure dropout layers are in test mode

        loader_length = len(data_loader)

        prediction = []  # 存储所有batch的output
        labels = []    # 存储所有batch的真实值

        prediction_all =  []
        labels_all = []

        criterion = nn.MSELoss().to(device)
        best_index = 0
        best_loss = np.inf

        for batch_index, batch_data in enumerate(data_loader):

            input, time_features, ground_truth = batch_data

            output = net(input.permute(0, 2, 1, 3), time_features)

            prediction.append(output.detach().reshape(-1).cpu().numpy())
            prediction_all.append(output.detach().cpu().numpy())
            labels.append(ground_truth.reshape(-1).cpu().numpy())
            labels_all.append(ground_truth.cpu().numpy())
            loss = criterion(output, ground_truth)

            if loss.item() < best_loss:
                best_index = batch_index
                best_loss = loss.item()

            data_dir = '04_{}_loss{}'.format(batch_index + 1, loss.item())   # 对每组预测数据创建一个文件夹，文件夹命名为组名+损失值
            dir = os.path.join(out_dir, data_dir)
            folder = os.path.exists(dir)
            if not folder:
                os.makedirs(dir)

            for i in range(6):    # 六个预测时刻
                Ground_truth = ground_truth[0, :, i].cpu().numpy()
                Prediction = output[0, :, i].detach().cpu().numpy()
                data = {'Ground_truth': Ground_truth, 'Prediction': Prediction}
                data = pd.DataFrame.from_dict(data)
                name = os.path.join(dir, str(i)+'.csv')             # 对每个时刻建立一个csv文件
                data.to_csv(name, index=False)

    print(best_index)

    prediction = np.array(prediction)
    prediction = np.reshape(prediction, -1)
    labels = np.array(labels)
    labels = np.reshape(labels, -1)

    ##################
    prediction_all = np.array(prediction_all).squeeze()
    labels_all = np.array(labels_all).squeeze()

    print(prediction_all.shape)
    print(labels_all.shape)

    np.save(os.path.join(out_dir, "prediction_all.npy"), prediction_all)
    np.save(os.path.join(out_dir, "labels_all.npy"), labels_all)

    num_samples, num_nodes, num_steps = prediction_all.shape

    np.random.seed(2025)

    # 随机选取节点索引
    selected_nodes = np.random.choice(num_nodes, size=min(5, num_nodes), replace=False)
    print(f"✅ 随机选取的节点索引: {selected_nodes.tolist()}")

    for t in range(num_steps):
        preds_t = prediction_all[:, selected_nodes, t]
        labels_t = labels_all[:, selected_nodes, t]

        # 合并真实值和预测值
        combined = np.concatenate([labels_t, preds_t], axis=1)

        # 生成列名
        columns = [f"node{n + 1}_true" for n in selected_nodes] + \
                  [f"node{n + 1}_pred" for n in selected_nodes]

        df = pd.DataFrame(combined, columns=columns)
        csv_path = os.path.join(out_dir, f"step_{t + 1}.csv")
        df.to_csv(csv_path, index=False)

    print(f"✅ 已生成 {num_steps} 个 CSV 文件，保存路径：{out_dir}")
    ########################

    MSE = metrics.mean_squared_error(labels, prediction)
    RMSE = metrics.mean_squared_error(labels, prediction)**0.5
    MAE = metrics.mean_absolute_error(labels, prediction)
    # MAPE = metrics.mean_absolute_percentage_error(labels, prediction)
    MAPE = mean_absolute_percentage_error(prediction, labels)

    print('MSE:{}, RMSE:{}, MAE:{}, MAPE:{}'.format(MSE, RMSE, MAE, MAPE))


def predict_main(net, data_loader, _mean, _std, type, out_dir, params_filename):

    print('load weight from:', params_filename)
    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results(net, data_loader, _mean, _std, type, out_dir)


# 对原始邻接矩阵进行处理
def process_adj(adj):
    adj = adj.numpy()
    std_data = []  # 收集所有具有记录距离的节点
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] > 1:
                std_data.append(adj[i, j])
    std_data = np.array(std_data)
    adj_mean = np.mean(std_data)
    adj_std = np.std(std_data)
    w = np.zeros((len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] > 0:
                w[i, j] = np.exp(-(adj[i, j] / adj_std) ** 2)
    return torch.Tensor(w)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__=='__main__':
    setup_seed(0)

    parser = argparse.ArgumentParser(description='STGAGRTN')
    parser.add_argument('--out_dir', type=str, default='./output_04')
    parser.add_argument('--dataset', type=str, default='./data/PEMS04/r1_d2_w2_PEMS04.npz', help='options: [./data/PEMS04/r1_d2_w2_PEMS04.npz, ./data/PEMS08/r1_d2_w2_PEMS08.npz]')
    parser.add_argument('--adj', type=str, default='./data/PEMS04/adj.csv', help='adjacency matrix, options: [./data/PEMS04/adj.csv, ./data/PEMS08/adj.csv]')
    parser.add_argument('--params_filename', type=str, default='./04.params', help='options: [./04.params, ./08.params]')

    args = parser.parse_args()

    out_dir = args.out_dir
    filename = args.dataset    ## Data generated by prepareData.py
    adj_mx = pd.read_csv(args.adj, header=None)
    params_filename = args.params_filename
    num_of_hours, num_of_days, num_of_weeks = 1, 1, 1 ## The same setting as prepareData.py
    
    # Training Hyparameter
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = DEVICE
    batch_size = 1   # 由于要输出每组预测结果，因此batch_size设为1
    in_channels = 1  # Channels of input
    embed_size = 64  # Dimension of hidden embedding features
    time_num = 288
    T_dim = 36  # Input length, should be the same as prepareData.py
    output_T_dim = 6  # Output Expected length
    heads = 2  # Number of Heads in MultiHeadAttention
    forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    dropout = 0.1

    folder = os.path.exists(out_dir)
    if not folder:
        os.makedirs(out_dir)
    
    ### Generate Data Loader
    train_loader, val_loader, test_loader, _mean, _std = load_data(filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)
    
    ### Adjacency Matrix Import
    adj_mx = np.array(adj_mx)
    A = adj_mx
    A = torch.Tensor(A)
    A = process_adj(A)

    ### Construct Network
    net = Model(
        A,
        in_channels,
        embed_size,
        time_num,
        T_dim,
        output_T_dim,
        heads,
        forward_expansion,
        dropout,
        307,
        10,
        device)
    net.to(device)

    predict_main(net, test_loader, _mean, _std, 'test', out_dir, params_filename)