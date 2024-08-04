import json
import numpy as np
import torch
from torch import optim
import random
from model import EncoderRNN, DecoderRNN, get_glove_embedding, device
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def load_dataset(file_path):
    with open(file_path, "r") as f:
        dataset = json.load(f)
    return dataset['X3'], dataset['Y3']

def load_jx_list(file_path):
    jx_list = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            num = line.split(" ")[0]
            jx_list.append(int(num))
    return jx_list

def init_models(hidden_size, embedding):
    encoder = EncoderRNN(hidden_size).to(device)
    decoder = DecoderRNN(embedding, hidden_size, 16).to(device)
    return encoder, decoder

def init_optimizers(models, learning_rate):
    return [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

def train_epoch(encoder, decoder, X, Y, jx_list, optimizers, criterion, batch_size):
    encoder.train()
    decoder.train()
    
    total_loss = 0
    for i in range(batch_size):
        index = random.randrange(0, len(X))
        x1 = torch.from_numpy(np.array(X[index])).to(device).float()
        y1 = torch.from_numpy(np.array(Y[index])).to(device).long()

        max_length = len(X[index])

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = 0

        encoder_hidden = encoder.initHidden()
        for ei in range(max_length):
            _, encoder_hidden = encoder(x1[ei], encoder_hidden)

        decoder_input = torch.tensor([[random.choice(jx_list)]], device=device)
        decoder_hidden = encoder_hidden
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            target = y1[di].view(-1)
            loss += criterion(decoder_output, target)
            decoder_input = target  # Teacher forcing

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        total_loss += loss.item() / max_length

        if i % 100 == 0:
            avg_loss = total_loss / (i+1)
            now_loss = loss.item() / max_length
            print(f"Batch: {i}, Avg Loss: {avg_loss:.4f}, Current Loss: {now_loss:.4f}")

    return total_loss / batch_size

def validate(encoder, decoder, X, Y, jx_list, criterion, batch_size):
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    with torch.no_grad():
        for i in range(batch_size):
            index = random.randrange(0, len(X))
            x1 = torch.from_numpy(np.array(X[index])).to(device).float()
            y1 = torch.from_numpy(np.array(Y[index])).to(device).long()

            max_length = len(X[index])

            loss = 0

            encoder_hidden = encoder.initHidden()
            for ei in range(max_length):
                _, encoder_hidden = encoder(x1[ei], encoder_hidden)

            decoder_input = torch.tensor([[random.choice(jx_list)]], device=device)
            decoder_hidden = encoder_hidden
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                target = y1[di].view(-1)
                loss += criterion(decoder_output, target)
                decoder_input = target  # Teacher forcing

            total_loss += loss.item() / max_length

    return total_loss / batch_size

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    hidden_size = 512
    learning_rate = 1e-3
    num_epochs = 20
    batch_size = 1000

    X, Y = load_dataset("dataset.json")
    jx_list = load_jx_list("glove/vocab2.txt")

    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    embedding = torch.nn.Embedding.from_pretrained(get_glove_embedding(16, "vectors2.txt"), freeze=False)
    encoder, decoder = init_models(hidden_size, embedding)
    optimizers = init_optimizers([encoder, decoder], learning_rate)
    scheduler = ReduceLROnPlateau(optimizers[0], 'min', patience=5)

    criterion = torch.nn.NLLLoss()

    writer = SummaryWriter()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, decoder, X_train, Y_train, jx_list, optimizers, criterion, batch_size)
        val_loss = validate(encoder, decoder, X_val, Y_val, jx_list, criterion, batch_size)

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizers': [opt.state_dict() for opt in optimizers],
        }, is_best)

    writer.close()

    # Load best model
    checkpoint = torch.load('model_best.pth.tar')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # Save best models
    torch.save(encoder, "checkpoints/ln_encoder.pth")
    torch.save(decoder, "checkpoints/ln_decoder.pth")

if __name__ == "__main__":
    main()