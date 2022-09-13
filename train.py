import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from baseline import Baseline
from sklearn.model_selection import train_test_split
import random
from process_data import pipline, read_data
import torch.nn.functional as F
from sklearn.metrics import f1_score


def get_labels(outputs,labels):
    # pred_labels 和 true_labels 便于后续计算F1分数
    outputs = F.softmax(outputs,dim=1).cpu().numpy()
    pred_labels = outputs.argmax(1)
    pred_labels = pred_labels.tolist()
    true_labels = labels.cpu().tolist()
    return pred_labels,true_labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    label_flat = labels.flatten()
    return np.sum(pred_flat == label_flat) / len(label_flat)


def train(model, config):
    idx2id = {}

    data_list = read_data('testA.json')
    for i, data in enumerate(data_list):
        data_id = data['id']
        idx2id[i] = data_id

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)

    epoch = config.epoch
    train_dataloader, val_dataloader = pipline(config.train_path, config.max_len, config.batch_size, is_train=True)
    total_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_loss = 10000
    loss_values = []

    for epoch_i in range(0, epoch):
        print('')
        print('=========== Epoch {:}/{:} =========='.format(epoch_i + 1, epoch))
        print('train...............')

        model.train()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        for i, batch in enumerate(train_dataloader):
            if i % 50 == 0 and i != 0:
                print('  Batch {:>5,}  of  {:>5,}'.format(i, len(train_dataloader)))
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(torch.int64).to(device)

            model.zero_grad()

            output = model(b_input_ids, b_masks)
            loss = criterion(output, b_labels)

            total_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        model.eval()
        val_loss, val_acc = 0, 0
        n_eval_steps = 0

        pred_labels = []
        true_labels = []
        for batch in val_dataloader:
            val_input_ids = batch[0].to(device)
            val_masks = batch[1].to(device)
            val_labels = batch[2].to(torch.int64).to(device)

            with torch.no_grad():
                outputs = model(val_input_ids, val_masks)
            loss = criterion(outputs, val_labels)
            batch_pred_labels, batch_true_labels = get_labels(outputs, val_labels)
            pred_labels += batch_pred_labels
            true_labels += batch_true_labels
            val_loss += loss.item()

            n_eval_steps += 1
            avg_val_loss = val_loss / len(val_dataloader)
            if avg_val_loss < best_loss:
                torch.save(model.state_dict(), "./models/baseline/model_parameter.pkl")
                best_loss = avg_val_loss
        epoch_score = f1_score(pred_labels, true_labels, average='macro')
        print("  F1: {0:.5f}".format(epoch_score))
        print("  val loss: {0:.5f}".format(avg_val_loss))
        print("  best val loss: {0:.5f}".format(best_loss))

    print("")
    print("Training complete!")