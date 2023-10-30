"""
Module for training the model.
"""

import math
import time

import tqdm
import torch


def training_loop(epochs, model, train_loader, valid_loader, criterion, optimizer, device, pad_idx, vocab_size):
    """Training loop."""
    history = {'epoch': 0, 'loss': [], 'ppl': [], 'acc': [], 'val-loss': [], 'val-ppl': [], 'val-acc': []}
    for epoch in range(1, epochs+1):
        start_epoch = time.time()

        # Training
        running_loss, running_total_seq, running_correct_preds, running_total_tokens = 0.0, 0, 0, 0
        model.train()
        for i, batch in enumerate(train_loader):  # NOTE: you can add tqdm(enumerate(train_loader)) to get a progress bar
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            target_ids = batch[1].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            running_loss += loss.item() * input_ids.shape[0]
            running_total_seq += input_ids.shape[0]
            mask_pad = input_ids != pad_idx
            running_correct_preds += torch.sum(torch.argmax(logits, dim=-1)[mask_pad] == target_ids[mask_pad]).item()
            running_total_tokens += torch.sum(mask_pad).item()

            loss.backward()
            optimizer.step()

            if i > 250:  # stop the epoch after 250 batches (simply to log more often when using lots of data)
                break

        train_loss = running_loss / running_total_seq
        train_acc = running_correct_preds / running_total_tokens

        # Validation
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device, pad_idx, vocab_size)

        # Logging
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        history['epoch'] += 1
        history['loss'].append(train_loss)
        history['ppl'].append(train_ppl)
        history['acc'].append(train_acc)
        history['val-loss'].append(val_loss)
        history['val-ppl'].append(val_ppl)
        history['val-acc'].append(val_acc)
        print(f"Epoch: {epoch}/{epochs} - loss={train_loss:.4f} - ppl={train_ppl:.4f} - acc={train_acc:.4f} - val-loss={val_loss:.4f} - val-ppl={val_ppl:.4f} - val-acc: {val_acc:.4f} ({time.time()-start_epoch:.2f}s/epoch)")
    print('Finished Training.')
    return history


def evaluate(model, loader, criterion, device, pad_idx, vocab_size):
    """Evaluate the model on the dataloader."""
    running_loss, running_correct_preds, running_total_tokens = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            target_ids = batch[1].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            running_loss += loss.item() * input_ids.shape[0]
            mask_pad = input_ids != pad_idx
            running_correct_preds += torch.sum(torch.argmax(logits, dim=-1)[mask_pad] == target_ids[mask_pad]).item()
            running_total_tokens += torch.sum(mask_pad).item()
            
    loss = running_loss / len(loader.dataset)
    acc = running_correct_preds / running_total_tokens
    return loss, acc