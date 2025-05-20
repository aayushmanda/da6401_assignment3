import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import re
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from model import EncoderRNN, DecoderRNN, Seq2Seq
from utils import TransliterationDataset, count_parameters, prepare_data



def train_epoch(model, dataloader, optimizer, criterion, device, train_lossi):
    model.train()
    epoch_loss = 0

    for batch_idx, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, trg)

        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Skip SOS token
        trg = trg[:, 1:].reshape(-1)  # Skip SOS token

        # Calculate loss
        loss = criterion(output, trg)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()
        train_lossi.append(loss.item())

    return epoch_loss / len(dataloader), train_lossi

# Updated evaluate
def evaluate(model, dataloader, criterion, device, val_lossi):
    model.eval()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0)

            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # Skip SOS token
            trg = trg[:, 1:].reshape(-1)  # Skip SOS token

            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            val_lossi.append(loss.item())

            # Calculate accuracy
            predictions = output.argmax(1)
            correct_mask = (predictions == trg) & (trg != 0)  # Exclude padding
            correct_predictions += correct_mask.sum().item()
            total_mask = trg != 0  # Exclude padding
            total_predictions += total_mask.sum().item()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return epoch_loss / len(dataloader), accuracy, val_lossi

def transliterate(model, src, input_lang, output_lang, device, max_length=30):
    model.eval()
    with torch.no_grad():
        # Reshape input to have batch size of 1: (seq_len, batch_size=1)
        src = src.unsqueeze(0)  # Add a batch dimension

        # Move to device
        src = src.to(device)

        # Initialize hidden state with batch_size=1
        batch_size = 1  # Explicitly set batch size to 1
        encoder_hidden = model.encoder.init_hidden(batch_size, device)  # Shape: (num_layers, 1, hidden_size)

        # Encoder forward pass
        encoder_outputs, encoder_hidden = model.encoder(src, encoder_hidden)

        # Decoder
        decoder_input = torch.tensor([[1]], device=device)  # SOS token
        decoder_hidden = encoder_hidden  # Shape: (num_layers, 1, hidden_size)
        output_chars = []

        for _ in range(max_length):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            char_idx = topi.item()

            if char_idx == 2:  # EOS token
                break

            output_chars.append(output_lang.index2char[char_idx])
            decoder_input = torch.tensor([[char_idx]], device=device)

        return ''.join(output_chars)

# Updated main function
def main():
    # Parameters
    batch_size = 64
    hidden_size = 256
    learning_rate = 0.001
    num_epochs = 5
    cell_type = "LSTM"
    num_layers = 2
    data_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    val_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    test_path = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    input_lang, output_lang, train_pairs = prepare_data(data_path)
    _, _, val_pairs = prepare_data(val_path, input_lang=input_lang, output_lang=output_lang)
    _, _, test_pairs = prepare_data(test_path, input_lang=input_lang, output_lang=output_lang)

    # Create datasets and dataloaders
    train_dataset = TransliterationDataset(data_path, input_lang, output_lang)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TransliterationDataset(val_path, input_lang, output_lang)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TransliterationDataset(test_path, input_lang, output_lang)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    encoder = EncoderRNN(input_lang.n_chars, hidden_size, num_layers, cell_type)
    decoder = DecoderRNN(hidden_size, output_lang.n_chars, num_layers, cell_type)
    model = Seq2SeqModel(encoder, decoder, device).to(device)
    count_parameters(model)

    # Define loss function and optimizer
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    train_lossi = []  # Per-batch training losses
    val_lossi = []    # Per-batch validation losses

    for epoch in range(num_epochs):
        # Train
        train_loss, train_lossi = train_epoch(model, train_loader, optimizer, criterion, device, train_lossi)
        # Validate
        val_loss, val_accuracy, val_lossi = evaluate(model, val_loader, criterion, device, val_lossi)

        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transliteration_model.pt")
            print("Model saved!")

        print("-----")

    # Plot per-batch training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_lossi, label='Training Loss ')
    plt.plot(val_lossi, label='Validation Loss ')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Per-Batch Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate on test set
    model.load_state_dict(torch.load("best_transliteration_model.pt"))
    test_loss, test_accuracy, _ = evaluate(model, test_loader, criterion, device, [])
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Example transliteration
    for i in range(min(10, len(test_dataset))):
        src, trg = test_dataset[i]
        src_text = ''.join([input_lang.index2char[idx.item()] for idx in src if idx.item() > 2])
        trg_text = ''.join([output_lang.index2char[idx.item()] for idx in trg if idx.item() > 2])

        pred_text = transliterate(model, src, input_lang, output_lang, device)

        print(f"Input: {src_text}")
        print(f"Target: {trg_text}")
        print(f"Prediction: {pred_text}")
        print("-----")

main()
