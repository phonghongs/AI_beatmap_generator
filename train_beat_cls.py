import json
import numpy as np
import torch
from torch import optim
import random
from model import EncoderRNN, OutPutLayer, device
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def load_dataset(file_path):
    with open(file_path, "r") as f:
        dataset = json.load(f)
    return dataset['X0'], dataset['Y0']

def init_models(hidden_size):
    encoder_f = EncoderRNN(hidden_size).to(device)
    encoder_b = EncoderRNN(hidden_size).to(device)
    cls_layer = OutPutLayer(hidden_size, 2).to(device)
    return encoder_f, encoder_b, cls_layer

def init_optimizers(models, learning_rate):
    return [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

def train_epoch(encoder_f, encoder_b, cls_layer, X, Y, optimizers, batch_size):
    encoder_f.train()
    encoder_b.train()
    cls_layer.train()
    
    total_loss = 0
    for i in range(batch_size):
        index = random.randrange(0, len(X))
        x0 = torch.from_numpy(np.array(X[index])).to(device).float()
        y0 = torch.from_numpy(np.array(Y[index])).to(device).long()

        max_length = len(X[index])

        for optimizer in optimizers:
            optimizer.zero_grad()

        encoder_f_outputs = torch.zeros(max_length, encoder_f.hidden_size, device=device)
        encoder_b_outputs = torch.zeros(max_length, encoder_b.hidden_size, device=device)

        encoder_f_hidden = encoder_f.initHidden()
        encoder_b_hidden = encoder_b.initHidden()

        for ei in range(max_length):
            encoder_output, encoder_f_hidden = encoder_f(x0[ei], encoder_f_hidden)
            encoder_f_outputs[ei] = encoder_output[0, 0]

            encoder_output, encoder_b_hidden = encoder_b(
                x0[max_length-ei-1], encoder_b_hidden)
            encoder_b_outputs[max_length-ei-1] = encoder_output[0, 0]

        encoder_outputs = torch.cat([encoder_f_outputs, encoder_b_outputs], dim=1)

        loss = 0
        for di in range(max_length):
            cls_input = encoder_outputs[di]
            cls_output = cls_layer(cls_input)
            target = y0[di].view(-1)
            loss += torch.nn.functional.nll_loss(cls_output, target)

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        total_loss += loss.item() / max_length

        if i % 100 == 0:
            avg_loss = total_loss / (i+1)
            now_loss = loss.item() / max_length
            print(f"Batch: {i}, Avg Loss: {avg_loss:.4f}, Current Loss: {now_loss:.4f}")

    return total_loss / batch_size

def validate(encoder_f, encoder_b, cls_layer, X, Y, batch_size):
    encoder_f.eval()
    encoder_b.eval()
    cls_layer.eval()
    
    total_loss = 0
    with torch.no_grad():
        for i in range(batch_size):
            index = random.randrange(0, len(X))
            x0 = torch.from_numpy(np.array(X[index])).to(device).float()
            y0 = torch.from_numpy(np.array(Y[index])).to(device).long()

            max_length = len(X[index])

            encoder_f_outputs = torch.zeros(max_length, encoder_f.hidden_size, device=device)
            encoder_b_outputs = torch.zeros(max_length, encoder_b.hidden_size, device=device)

            encoder_f_hidden = encoder_f.initHidden()
            encoder_b_hidden = encoder_b.initHidden()

            for ei in range(max_length):
                encoder_output, encoder_f_hidden = encoder_f(x0[ei], encoder_f_hidden)
                encoder_f_outputs[ei] = encoder_output[0, 0]

                encoder_output, encoder_b_hidden = encoder_b(
                    x0[max_length-ei-1], encoder_b_hidden)
                encoder_b_outputs[max_length-ei-1] = encoder_output[0, 0]

            encoder_outputs = torch.cat([encoder_f_outputs, encoder_b_outputs], dim=1)

            loss = 0
            for di in range(max_length):
                cls_input = encoder_outputs[di]
                cls_output = cls_layer(cls_input)
                target = y0[di].view(-1)
                loss += torch.nn.functional.nll_loss(cls_output, target)

            total_loss += loss.item() / max_length

    return total_loss / batch_size

def save_checkpoint(state, is_best, filename='checkpoint_beat.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_beat.pth.tar')

def main():
    hidden_size = 512
    learning_rate = 1e-3
    num_epochs = 20
    batch_size = 10000

    X, Y = load_dataset("dataset.json")
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    encoder_f, encoder_b, cls_layer = init_models(hidden_size)
    optimizers = init_optimizers([encoder_f, encoder_b, cls_layer], learning_rate)
    scheduler = ReduceLROnPlateau(optimizers[0], 'min', patience=5)

    writer = SummaryWriter()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder_f, encoder_b, cls_layer, X_train, Y_train, optimizers, batch_size)
        val_loss = validate(encoder_f, encoder_b, cls_layer, X_val, Y_val, batch_size)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'encoder_f': encoder_f.state_dict(),
            'encoder_b': encoder_b.state_dict(),
            'cls_layer': cls_layer.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizers': [opt.state_dict() for opt in optimizers],
        }, is_best)

    writer.close()

    # Load best model
    checkpoint = torch.load('model_best_beat.pth.tar')
    encoder_f.load_state_dict(checkpoint['encoder_f'])
    encoder_b.load_state_dict(checkpoint['encoder_b'])
    cls_layer.load_state_dict(checkpoint['cls_layer'])

    # Save best models
    torch.save(encoder_f, "checkpoints/encoder_beat1.pth")
    torch.save(encoder_b, "checkpoints/encoder_beat2.pth")
    torch.save(cls_layer, "checkpoints/beat_cls_layer.pth")

if __name__ == "__main__":
    main()