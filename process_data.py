import json

import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
from transformers import BertTokenizer
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


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


def load_data(data_list, tokenizer, MAX_LEN, is_train=True):
    '''
    讲json的list处理成Bert中的token_idx
    :param data_list: list
    :param is_train: Boolean
    :return:
    '''
    input_ids = []
    label_list = []
    for data in data_list:
        data_id = data['id']
        title = data['title']
        assignee = data['assignee']
        abstract = data['abstract']
        if is_train is True:
            label = data['label_id']
        else:
            label = '[UNK]'
        label_list.append(label)
        text = "这份专利的标题为：《{}》，由“{}”公司申请，详细说明如下：{}".format(title, assignee, abstract)
        encoded_text = tokenizer.encode(
            text,
            add_special_tokens=True
        )
        input_ids.append(encoded_text)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
    input_pairs = []
    X = []
    y = []
    for i in range(len(input_ids)):
        X.append(input_ids[i])
        y.append(label_list[i])
    X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), test_size=0.1, random_state=42)

    return X_train, X_val, y_train, y_val


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


def build_dataloader(X, y, attention_masks, batch_size, is_train=True):
    '''
    生成dataloader
    :param input_pairs:
    :param attention_masks:
    :param batch_size:
    :param is_train:
    :return:
    '''
    input_ids = torch.tensor(X)
    labels = torch.tensor(y)
    data = TensorDataset(input_ids, attention_masks, labels)
    if is_train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def pipline(data_path, MAX_LEN, batch_size, is_train=True):
    data_list = read_data(data_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    X_train, X_val, y_train, y_val = load_data(data_list=data_list, tokenizer=tokenizer, is_train=is_train, MAX_LEN=MAX_LEN)
    train_masks = generate_mask(X_train)
    val_masks = generate_mask(X_val)
    train_dataloader = build_dataloader(X=X_train, y=y_train, attention_masks=train_masks, batch_size=batch_size,
                                  is_train=is_train)
    val_dataloader = build_dataloader(X=X_val, y=y_val, attention_masks=val_masks, batch_size=batch_size,
                                        is_train=is_train)

    return train_dataloader, val_dataloader

