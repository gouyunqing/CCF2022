from baseline import Baseline
import torch
import json
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
from transformers import BertTokenizer
import pandas as pd
from config import Config
import torch.nn as nn
import torch.nn.functional as F


def read_data(data_path):
    '''
    读入json数据，讲每一条json存入列表中并返回
    :param data_path: 数据路径
    :return: 返回一个字典的list
    '''
    data_list = []
    with open(data_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data_list.append(data)
    return data_list


def load_data(data_list, tokenizer, MAX_LEN):
    '''
    讲json的list处理成Bert中的token_idx
    :param data_list: list
    :param is_train: Boolean
    :return:
    '''
    input_ids = []
    id_list = []
    for data in data_list:
        data_id = data['id']
        title = data['title']
        assignee = data['assignee']
        abstract = data['abstract']

        id_list.append(data_id)
        text = title + abstract
        encoded_text = tokenizer.encode(
            text,
            add_special_tokens=True
        )
        input_ids.append(encoded_text)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
    X = []
    for i in range(len(input_ids)):
        X.append(input_ids[i])

    return X, id_list


def generate_mask(X):
    '''
    为每个输入向量生成一个mask向量。所有是padding的位置为False, 所有text位置为True
    :param input_pairs:
    :return:mask向量的列表
    '''
    attention_masks = []
    for x in X:
        mask = [int(_) > 0 for _ in x]
        attention_masks.append(mask)

    return torch.tensor(attention_masks)


def build_dataloader(X,attention_masks, batch_size):
    '''
    生成dataloader
    :param input_pairs:
    :param attention_masks:
    :param batch_size:
    :param is_train:
    :return:
    '''
    input_ids = torch.tensor(X)
    data = TensorDataset(input_ids, attention_masks)
    sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def pipline(data_path, MAX_LEN, batch_size):
    data_list = read_data(data_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    X, id_list = load_data(data_list=data_list, tokenizer=tokenizer, MAX_LEN=MAX_LEN)
    masks = generate_mask(X)
    dataloader = build_dataloader(X=X, attention_masks=masks, batch_size=batch_size)

    return dataloader, id_list


if __name__ == '__main__':
    config = Config()
    model = Baseline()
    model.load_state_dict(torch.load('./models/baseline/model_parameter.pkl'))
    dataloader, id_list = pipline(config.test_path, config.max_len, config.batch_size)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    n_eval_steps = 0
    pred_labels = []
    for batch in dataloader:
        input_ids = batch[0].to(device)
        masks = batch[1].to(device)

        with torch.no_grad():
            outputs = model(input_ids, masks)
        outputs = F.softmax(outputs, dim=1).cpu().numpy()
        print(outputs)
        batch_pred_labels = outputs.argmax(1)
        batch_pred_labels = batch_pred_labels.tolist()
        print(batch_pred_labels)

        pred_labels += batch_pred_labels

    df = pd.DataFrame({'id': id_list, 'label': pred_labels})
    df.to_csv('submit_baseline.csv', index=None)