import numpy as np
from numpy import ma
import pandas as pd
from sklearn.utils import check_random_state
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn.utils import prune
import torch.nn.functional as F
import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TabularDataset(Dataset):
    def __init__(self, X, y):
        """
        Characterizes a Dataset for PyTorch
        """
        
        self.n = X.shape[0]
        
        self.y = y.astype(np.float32).values.reshape(-1, 1)
        self.X = X.astype(np.float32).values
    
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.X[idx]]



class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size, drop):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.drop = drop
        
        # First Layer
        first_lin_layer = nn.Linear(self.input_size, self.hidden_size)
        
        # Hidden Layers
        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(self.hidden_size, self.hidden_size)
                for i in range(self.hidden_layers - 1)
            ]
        )
        
        # Output Layer
        self.output_layer = nn.Linear(self.hidden_size, output_size)
        
        # Dropout Layers
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(self.drop) for layer in self.lin_layers]
        )
        
    def forward(self, x):
        
        for lin, d in zip(self.lin_layers, self.droput_layers):
            x = F.relu(lin(x))
            x = d(x)
        x = self.output_layer(x)
        return x




class DistFCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size, drop):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.drop = drop
        
        # First Layer
        first_lin_layer = nn.Linear(self.input_size, self.hidden_size)
        
        # Hidden Layers
        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(self.hidden_size, self.hidden_size)
                for i in range(self.hidden_layers - 1)
            ]
        )
        
        # Output mean Layer
        self.output_mean_layer = nn.Linear(self.hidden_size, 1)
        
        # Output var Layer
        self.output_sd_layer = nn.Linear(self.hidden_size, 1)
        
        # Dropout Layers
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(self.drop) for layer in self.lin_layers]
        )
        
    def forward(self, x):
        
        for lin, d in zip(self.lin_layers, self.droput_layers):
            x = F.relu(lin(x))
            x = d(x)
        mean = self.output_mean_layer(x)
        sd = F.relu(self.output_sd_layer(x))
        return mean, sd



def PICP(y_true, low, upp):
    inside = 0
    y_true = y_true.ravel()
    for i in range(0, len(y_true)):
        if low[i] <= y_true[i] <= upp[i]:
            inside += 1
    picp = inside / len(y_true)
    return picp


def AIW(low, upp, y_true):
    width = upp - low
    width_mean = width.mean()
    #width_sc = width_mean/(y_true.max() - y_true.min()) 
    return width_mean


def extract_layer(weight, bias, l):
    df_sub = pd.DataFrame(weight[l]).add_prefix('node_')
    df_sub['intercept'] = bias[l]
    df_sub['layer'] = l
    df_sub['node'] = range(len(df_sub))
    return df_sub

def constraint_extrapolation_MLP(weight, bias, names):
    n_layers = len(names)
    constraints = pd.concat([extract_layer(weight, bias, l) for l in range(n_layers)],axis=0)
    cols_to_move = ['intercept', 'layer', 'node']
    constraints = constraints[cols_to_move + [col for col in constraints.columns if col not in cols_to_move]]
    return constraints


def cyclical_encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def lag_gen(data, col, lag_list):
    for i in lag_list:
        data[col + f'_{i}lag'] = data[col].shift(i)
        data = data.copy()
    return data


def lag_pert(data, col, lag_list, error):
    for i in lag_list:
        data[col + f'_{i}lag'] = data[col + f'_{i}lag']*np.random.uniform(low=1-error, high=1+error)
        data = data.copy()
    return data


def train_test_split(X, y, train_ratio=0.7):
    num_ts, num_periods, num_features = X.shape
    train_periods = int(num_periods * train_ratio)
    random.seed(2)
    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, train_periods:, :]
    yte = y[:, train_periods:]
    return Xtr, ytr, Xte, yte


class StandardScaler:
    
    def fit_transform(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std


