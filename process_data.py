import json
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
from process_data import read_data
from transformers import BertTokenizer
import random


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
    input_pairs = []
    for data in data_list:
        data_id = data['id']
        title = data['title']
        assignee = data['assignee']
        abstract = data['abstract']
        if is_train is True:
            label = data['label_id']
        else:
            label = '[UNK]'

        text = title + abstract
        encoded_text = tokenizer.encode(
            text,
            add_special_tokens=True
        )
        input_pairs.append(encoded_text)
    input_pairs = pad_sequences(input_pairs, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
    for i in range(len(input_pairs)):
        input_pairs[i] = [input_pairs[i], label]

    return input_pairs


def generate_mask(input_pairs):
    '''
    为每个输入向量生成一个mask向量。所有是padding的位置为False, 所有text位置为True
    :param input_pairs:
    :return:mask向量的列表
    '''
    attention_masks = []
    for pair in input_pairs:
        input_id = pair[0]
        mask = [int(_) > 0 for _ in input_id]
        attention_masks.append(mask)

    return attention_masks


def build_dataloader(input_pairs, attention_masks, batch_size, is_train=True):
    '''
    生成dataloader
    :param input_pairs:
    :param attention_masks:
    :param batch_size:
    :param is_train:
    :return:
    '''
    input_ids = []
    labels = []
    for pair in input_pairs:
        input_ids.append(pair[0])
        labels.append(pair[1])

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
    input_pairs = load_data(data_list=data_list, tokenizer=tokenizer, is_train=is_train, MAX_LEN=MAX_LEN)
    attention_masks = generate_mask(input_pairs)
    dataloader = build_dataloader(input_pairs=input_pairs, attention_masks=attention_masks, batch_size=batch_size,
                                  is_train=is_train)

    return dataloader

