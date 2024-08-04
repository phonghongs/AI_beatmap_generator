# data_utils.py

import json
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path, key_x, key_y, batch_size=32, train_ratio=0.8):
    with open(file_path, "r") as f:
        dataset = json.load(f)
    
    X = dataset[key_x]
    Y = dataset[key_y]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def load_jx_list(file_path):
    jx_list = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            num = line.split(" ")[0]
            jx_list.append(int(num))
    return jx_list
