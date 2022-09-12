import json


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


def load_data(data_list, tokenizer, is_train=True):
    '''
    讲json的list处理成Bert中的token_idx
    :param data_list: list
    :param is_train: Boolean
    :return: 一个list，每个元素都是一个长度为1（test）或长度为2（train）的list， [0]位置是token_idx的序列， [1]位置是label
    '''
    for data in data_list:
        data_id = data['id']
        title = data['title']
        assignee = data['assignee']
        abstract = data['abstract']
        if is_train is True:
            label = data['label_id']


